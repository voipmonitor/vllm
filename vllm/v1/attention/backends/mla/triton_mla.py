# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import (
    _decode_softmax_reducev_fwd,
    _fwd_grouped_kernel_stage1,
)

try:
    from vllm.v1.attention.backends.mla.triton_mla_tuning import (
        lookup_config as _lookup_tuned_config,
    )
except ImportError:
    _lookup_tuned_config = None

logger = init_logger(__name__)

MAX_NUM_KV_SPLITS = 64
CG_NUM_KV_SPLITS = 64


def _pick_num_kv_splits(B: int, q_num_heads: int) -> int:
    """Pick a CUDA-graph-safe split count for a decode bucket."""
    h_blocks = max(1, (q_num_heads + 7) // 8)
    target = max(1, 576 // (B * h_blocks))
    p = 1
    while (p << 1) <= target:
        p <<= 1
    return max(1, min(MAX_NUM_KV_SPLITS, p))


_SHARED_CG_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_shared_cg_buffer(
    key_prefix: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    init_zeros: bool = False,
) -> torch.Tensor:
    key = (key_prefix, device, shape, dtype)
    buf = _SHARED_CG_BUFFERS.get(key)
    if buf is None:
        if init_zeros:
            buf = torch.zeros(shape, dtype=dtype, device=device)
        else:
            buf = torch.empty(shape, dtype=dtype, device=device)
        _SHARED_CG_BUFFERS[key] = buf
    return buf


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("supports_dcp_with_varlen", True)
        super().__init__(*args, **kwargs)

        max_num_tokens = self.vllm_config.scheduler_config.max_num_seqs * max(
            1, int(self.reorder_batch_threshold)
        )
        block_size = self.kv_cache_spec.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        max_blocks_per_req = (max_model_len + block_size - 1) // block_size

        self._cg_buf_block_table: torch.Tensor | None = None
        self._cg_buf_seq_lens: torch.Tensor | None = None
        self._cg_max_num_tokens = max_num_tokens
        self._cg_max_blocks_per_req = max_blocks_per_req

    def _maybe_lazy_init_cg_bufs(
        self,
        device: torch.device,
        block_table_dtype: torch.dtype,
        seq_lens_dtype: torch.dtype,
    ) -> None:
        if self._cg_buf_block_table is None:
            self._cg_buf_block_table = torch.zeros(
                (self._cg_max_num_tokens, self._cg_max_blocks_per_req),
                dtype=block_table_dtype,
                device=device,
            )
            self._cg_buf_seq_lens = torch.zeros(
                (self._cg_max_num_tokens,),
                dtype=seq_lens_dtype,
                device=device,
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> MLACommonDecodeMetadata:
        num_reqs = seq_lens_device.shape[0]
        if num_decode_tokens > num_reqs and num_reqs > 0:
            qpr = num_decode_tokens // num_reqs
            assert num_decode_tokens == num_reqs * qpr, (
                "TritonMLA decode-path expects uniform query_len per request; "
                f"got num_decode_tokens={num_decode_tokens}, num_reqs={num_reqs}"
            )

            self._maybe_lazy_init_cg_bufs(
                device=block_table_tensor.device,
                block_table_dtype=block_table_tensor.dtype,
                seq_lens_dtype=seq_lens_device.dtype,
            )
            assert self._cg_buf_block_table is not None
            assert self._cg_buf_seq_lens is not None

            bt_rows = num_decode_tokens
            bt_cols_src = block_table_tensor.shape[1]
            assert bt_rows <= self._cg_buf_block_table.shape[0], (
                f"spec-verify num_decode_tokens={bt_rows} exceeds CG buffer "
                f"capacity {self._cg_buf_block_table.shape[0]}"
            )
            assert bt_cols_src <= self._cg_buf_block_table.shape[1], (
                f"block_table has {bt_cols_src} columns > CG buffer "
                f"{self._cg_buf_block_table.shape[1]}"
            )

            for j in range(qpr):
                self._cg_buf_block_table[j:bt_rows:qpr, :bt_cols_src].copy_(
                    block_table_tensor
                )

            arange = torch.arange(
                qpr,
                device=seq_lens_device.device,
                dtype=seq_lens_device.dtype,
            )
            if dcp_tot_seq_lens_device is not None and self.dcp_world_size > 1:
                from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

                global_per_query = (
                    dcp_tot_seq_lens_device.to(seq_lens_device.dtype).unsqueeze(1)
                    - (qpr - 1)
                    + arange
                ).reshape(-1)
                expanded = get_dcp_local_seq_lens(
                    global_per_query,
                    self.dcp_world_size,
                    self.dcp_rank,
                    self.cp_kv_cache_interleave_size,
                )
            else:
                expanded = (
                    seq_lens_device.unsqueeze(1) - (qpr - 1) + arange
                ).reshape(-1)
            self._cg_buf_seq_lens[:bt_rows].copy_(expanded)

            if dcp_tot_seq_lens_device is not None:
                dcp_tot_expanded = (
                    dcp_tot_seq_lens_device.unsqueeze(1) - (qpr - 1) + arange
                ).reshape(-1)
            else:
                dcp_tot_expanded = None

            return MLACommonDecodeMetadata(
                block_table=self._cg_buf_block_table[:bt_rows],
                seq_lens=self._cg_buf_seq_lens[:bt_rows],
                dcp_tot_seq_lens=dcp_tot_expanded,
            )

        return MLACommonDecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()
        self._use_tuned_config = (
            current_platform.is_cuda()
            and current_platform.has_device_capability(120)
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and _lookup_tuned_config is not None
        )

        vllm_cfg = get_current_vllm_config_or_none()
        if vllm_cfg is not None:
            scheduler_cfg = vllm_cfg.scheduler_config
            spec_cfg = vllm_cfg.speculative_config
            qpr_max = 1 + (
                spec_cfg.num_speculative_tokens
                if spec_cfg is not None and spec_cfg.num_speculative_tokens is not None
                else 0
            )
            self._cg_max_tokens = scheduler_cfg.max_num_seqs * qpr_max
            cg_max = getattr(
                vllm_cfg.compilation_config,
                "max_cudagraph_capture_size",
                None,
            )
            if cg_max is not None:
                self._cg_max_tokens = min(self._cg_max_tokens, cg_max)
            self._tuning_max_model_len = vllm_cfg.model_config.max_model_len
        else:
            self._cg_max_tokens = 512
            self._tuning_max_model_len = 262144

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]

        if envs.VLLM_BATCH_INVARIANT:
            kernel_cfg = {
                "num_kv_splits": 1,
                "BLOCK_N": 32,
                "BLOCK_H": 8,
                "num_stages": 2,
                "num_warps": 4,
            }
        else:
            kernel_cfg = None
            if self._use_tuned_config:
                kernel_cfg = _lookup_tuned_config(
                    q_num_heads, self._tuning_max_model_len, B
                )
            if kernel_cfg is None:
                kernel_cfg = {
                    "num_kv_splits": _pick_num_kv_splits(B, q_num_heads),
                    "BLOCK_N": 32,
                    "BLOCK_H": 8,
                    "num_stages": 2,
                    "num_warps": 4,
                }
        num_kv_splits = kernel_cfg["num_kv_splits"]

        assert B <= self._cg_max_tokens, (
            f"forward_mqa: B={B} exceeds CG capture max {self._cg_max_tokens}"
        )
        o_buf = _get_shared_cg_buffer(
            "o",
            (self._cg_max_tokens, q_num_heads, self.kv_lora_rank),
            q.dtype,
            q.device,
        )
        lse_buf = _get_shared_cg_buffer(
            "lse",
            (self._cg_max_tokens, q_num_heads),
            q.dtype,
            q.device,
        )
        attn_logits_buf = _get_shared_cg_buffer(
            "attn_logits",
            (
                self._cg_max_tokens,
                q_num_heads,
                MAX_NUM_KV_SPLITS,
                self.kv_lora_rank + 1,
            ),
            torch.float32,
            q.device,
        )
        o = o_buf[:B]
        lse = lse_buf[:B]
        attn_logits = attn_logits_buf[:B, :, :num_kv_splits, :]

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        Lk = kv_c_and_k_pe_cache.shape[-1]
        Lv = kv_c_cache.shape[-1]
        BLOCK_DV = triton.next_power_of_2(Lv)
        kv_group_num = q_num_heads
        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        grid_s1 = (
            B,
            triton.cdiv(q_num_heads, min(kernel_cfg["BLOCK_H"], kv_group_num)),
            num_kv_splits,
        )
        _fwd_grouped_kernel_stage1[grid_s1](
            q,
            kv_c_and_k_pe_cache,
            kv_c_and_k_pe_cache,
            self.scale,
            block_table,
            seq_lens,
            attn_logits,
            block_table.stride(0),
            q.stride(0),
            q.stride(1),
            kv_c_and_k_pe_cache.stride(-3),
            kv_c_and_k_pe_cache.stride(-2),
            kv_c_and_k_pe_cache.stride(-3),
            kv_c_and_k_pe_cache.stride(-2),
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            layer._k_scale,
            layer._k_scale,
            kv_group_num=kv_group_num,
            q_head_num=q_num_heads,
            BLOCK_DMODEL=self.kv_lora_rank,
            BLOCK_DPE=self.qk_rope_head_dim,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=kernel_cfg["BLOCK_N"],
            BLOCK_H=kernel_cfg["BLOCK_H"],
            NUM_KV_SPLITS=num_kv_splits,
            PAGE_SIZE=PAGE_SIZE,
            logit_cap=0.0,
            num_warps=kernel_cfg["num_warps"],
            num_stages=kernel_cfg["num_stages"],
            Lk=Lk,
            Lv=Lv,
            IS_MLA=True,
        )
        _decode_softmax_reducev_fwd(
            attn_logits,
            q,
            o,
            lse,
            kv_c_cache,
            seq_lens,
            num_kv_splits,
        )

        return o, lse
