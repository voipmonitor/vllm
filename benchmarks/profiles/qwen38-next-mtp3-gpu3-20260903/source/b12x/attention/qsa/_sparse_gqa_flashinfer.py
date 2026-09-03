"""Research-only FlashInfer qualification path for QSA sparse decode.

The QSA selector emits arbitrary logical token positions, while FlashInfer's
TRT-LLM/XQA decode kernel consumes dense fixed-size pages.  This module packs
the selected paged K/V rows into caller-stable process-owned qualification
buffers, then runs the FlashInfer decode kernel.  It is intentionally limited
to measuring whether a packed-kernel design can outperform B12X's split/merge
core; production integration must move these buffers into the planned QSA
binding before this path can be supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import torch
import triton
import triton.language as tl

_PACKED_PAGE_SIZE = 64
_WORKSPACE_BYTES = 128 * 1024 * 1024


@triton.jit
def _count_selected_positions(
    selected_positions,
    request_ids,
    query_positions,
    counts,
    selected_stride,
    selection_width: tl.constexpr,
    block_width: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_width)
    request = tl.load(request_ids + row).to(tl.int64)
    query_position = tl.load(query_positions + row).to(tl.int64)
    valid_request = request >= 0
    positions = tl.load(
        selected_positions + row * selected_stride + columns,
        mask=columns < selection_width,
        other=-1,
    ).to(tl.int64)
    valid = (
        valid_request
        & (positions >= 0)
        & (positions <= query_position)
    )
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _pack_selected_paged_kv(
    key_cache,
    value_cache,
    block_table,
    request_ids,
    query_positions,
    selected_positions,
    packed_key,
    packed_value,
    num_cache_pages,
    num_requests,
    table_width,
    key_page_stride: tl.constexpr,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    value_page_stride: tl.constexpr,
    value_token_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    table_row_stride: tl.constexpr,
    selected_row_stride: tl.constexpr,
    packed_row_stride: tl.constexpr,
    page_size: tl.constexpr,
    selection_width: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_tokens: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    token_block = tl.program_id(2)
    columns = token_block * block_tokens + tl.arange(0, block_tokens)
    dimensions = tl.arange(0, head_dim)
    request = tl.load(request_ids + row).to(tl.int64)
    query_position = tl.load(query_positions + row).to(tl.int64)
    valid_request = (request >= 0) & (request < num_requests)
    positions = tl.load(
        selected_positions + row * selected_row_stride + columns,
        mask=columns < selection_width,
        other=-1,
    ).to(tl.int64)
    logical_pages = positions // page_size
    offsets_in_page = positions - logical_pages * page_size
    valid = (
        valid_request
        & (columns < selection_width)
        & (positions >= 0)
        & (positions <= query_position)
        & (logical_pages >= 0)
        & (logical_pages < table_width)
    )
    physical_pages = tl.load(
        block_table + request * table_row_stride + logical_pages,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    valid &= (physical_pages >= 0) & (physical_pages < num_cache_pages)
    key_offsets = (
        physical_pages[:, None] * key_page_stride
        + offsets_in_page[:, None] * key_token_stride
        + head * key_head_stride
        + dimensions[None, :]
    )
    value_offsets = (
        physical_pages[:, None] * value_page_stride
        + offsets_in_page[:, None] * value_token_stride
        + head * value_head_stride
        + dimensions[None, :]
    )
    output_offsets = (
        (row * packed_row_stride + columns)[:, None] * kv_heads * head_dim
        + head * head_dim
        + dimensions[None, :]
    )
    mask = valid[:, None]
    tl.store(
        packed_key + output_offsets,
        tl.load(key_cache + key_offsets, mask=mask, other=0.0),
        mask=mask,
    )
    tl.store(
        packed_value + output_offsets,
        tl.load(value_cache + value_offsets, mask=mask, other=0.0),
        mask=mask,
    )


@dataclass(frozen=True)
class _Buffers:
    packed_key: torch.Tensor
    packed_value: torch.Tensor
    workspace: torch.Tensor
    block_tables: torch.Tensor
    sequence_lengths: torch.Tensor
    packed_row_stride: int


_BUFFER_LOCK = Lock()
_BUFFERS: dict[tuple[object, ...], _Buffers] = {}
_SCALES: dict[tuple[int, float], torch.Tensor] = {}


def _scaled_k_descale(
    descale: torch.Tensor | None,
    softmax_scale: float,
) -> float | torch.Tensor:
    if descale is None:
        return float(softmax_scale)
    key = (descale.data_ptr(), float(softmax_scale))
    with _BUFFER_LOCK:
        cached = _SCALES.get(key)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer QSA scale must be prepared before CUDA graph capture"
            )
        cached = torch.empty_like(descale)
        torch.mul(descale, float(softmax_scale), out=cached)
        _SCALES[key] = cached
        return cached


def _buffers(
    *,
    rows: int,
    selection_width: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _Buffers:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, rows, selection_width, kv_heads, head_dim, dtype)
    with _BUFFER_LOCK:
        cached = _BUFFERS.get(key)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer QSA qualification buffers must be allocated before "
                "CUDA graph capture"
            )
        pages_per_row = triton.cdiv(selection_width, _PACKED_PAGE_SIZE)
        packed_row_stride = pages_per_row * _PACKED_PAGE_SIZE
        packed_shape = (rows * packed_row_stride, kv_heads, head_dim)
        packed_key = torch.empty(packed_shape, dtype=dtype, device=device)
        packed_value = torch.empty_like(packed_key)
        block_tables = (
            torch.arange(rows * pages_per_row, dtype=torch.int32, device=device)
            .view(rows, pages_per_row)
            .contiguous()
        )
        cached = _Buffers(
            packed_key=packed_key,
            packed_value=packed_value,
            workspace=torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
            block_tables=block_tables,
            sequence_lengths=torch.empty(rows, dtype=torch.int32, device=device),
            packed_row_stride=packed_row_stride,
        )
        _BUFFERS[key] = cached
        return cached


def launch_sparse_paged_gqa_flashinfer(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the packed FlashInfer qualification path into ``output``."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    rows, _, head_dim = map(int, query.shape)
    kv_heads = int(key_cache.shape[2])
    selection_width = int(selected_positions.shape[1])
    buffers = _buffers(
        rows=rows,
        selection_width=selection_width,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=key_cache.dtype,
        device=query.device,
    )
    _count_selected_positions[(rows,)](
        selected_positions,
        request_ids,
        query_positions,
        buffers.sequence_lengths,
        int(selected_positions.stride(0)),
        selection_width=selection_width,
        block_width=triton.next_power_of_2(selection_width),
        num_warps=8,
    )
    _pack_selected_paged_kv[
        (rows, kv_heads, triton.cdiv(selection_width, 16))
    ](
        key_cache,
        value_cache,
        block_table,
        request_ids,
        query_positions,
        selected_positions,
        buffers.packed_key,
        buffers.packed_value,
        int(key_cache.shape[0]),
        int(block_table.shape[0]),
        int(block_table.shape[1]),
        key_page_stride=int(key_cache.stride(0)),
        key_token_stride=int(key_cache.stride(1)),
        key_head_stride=int(key_cache.stride(2)),
        value_page_stride=int(value_cache.stride(0)),
        value_token_stride=int(value_cache.stride(1)),
        value_head_stride=int(value_cache.stride(2)),
        table_row_stride=int(block_table.stride(0)),
        selected_row_stride=int(selected_positions.stride(0)),
        packed_row_stride=buffers.packed_row_stride,
        page_size=int(key_cache.shape[1]),
        selection_width=selection_width,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_tokens=16,
        num_warps=8,
    )
    pages_per_row = buffers.packed_row_stride // _PACKED_PAGE_SIZE
    key_pages = (
        buffers.packed_key.view(
            rows * pages_per_row,
            _PACKED_PAGE_SIZE,
            kv_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
    )
    value_pages = (
        buffers.packed_value.view(
            rows * pages_per_row,
            _PACKED_PAGE_SIZE,
            kv_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
    )
    trtllm_batch_decode_with_kv_cache(
        query=query,
        kv_cache=(key_pages, value_pages),
        workspace_buffer=buffers.workspace,
        block_tables=buffers.block_tables,
        seq_lens=buffers.sequence_lengths,
        max_seq_len=buffers.packed_row_stride,
        bmm1_scale=_scaled_k_descale(k_descale, softmax_scale),
        bmm2_scale=1.0 if v_descale is None else v_descale,
        out=output[:rows],
        kv_layout="HND",
        enable_pdl=True,
        backend="auto",
    )
    return output[:rows]


__all__ = ["launch_sparse_paged_gqa_flashinfer"]
