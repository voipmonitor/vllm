# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch
from flashinfer import BatchMLAPagedAttentionWrapper

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
    get_mla_dims,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024

_shared_workspaces: dict[torch.device, torch.Tensor] = {}


def _get_shared_workspace(device: torch.device) -> torch.Tensor:
    workspace = _shared_workspaces.get(device)
    if workspace is None:
        workspace = torch.zeros(
            FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _shared_workspaces[device] = workspace
    return workspace


def _create_wrapper(
    workspace: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_pages_per_req: int,
    tokens_per_req: int = 1,
) -> BatchMLAPagedAttentionWrapper:
    """Create a CUDA-graph-safe FlashInfer MLA wrapper for one batch shape."""
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(
        batch_size * max_pages_per_req,
        dtype=torch.int32,
        device=device,
    )
    kv_len_arr = torch.zeros(batch_size, dtype=torch.int32, device=device)
    return BatchMLAPagedAttentionWrapper(
        workspace,
        use_cuda_graph=True,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_arr,
        backend="auto",
    )


@dataclass
class FlashInferMLADecodeMetadata(MLACommonDecodeMetadata):
    wrapper: BatchMLAPagedAttentionWrapper | None = None


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata[FlashInferMLADecodeMetadata]):
    pass


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[FlashInferMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
        **kwargs,
    ):
        max_model_len = vllm_config.model_config.max_model_len
        block_size = kv_cache_spec.block_size
        self._max_pages_per_req = (max_model_len + block_size - 1) // block_size
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            supports_dcp_with_varlen=True,
            metadata_cls=FlashInferMLAMetadata,
            **kwargs,
        )

        mla_dims = get_mla_dims(self.model_config)
        self._kv_lora_rank = mla_dims.kv_lora_rank
        self._qk_nope_head_dim = mla_dims.qk_nope_head_dim
        self._qk_rope_head_dim = mla_dims.qk_rope_head_dim
        self._page_size = kv_cache_spec.block_size
        self._device = device
        self._shared_workspace = _get_shared_workspace(device)
        self._wrappers: dict[tuple[int, int], BatchMLAPagedAttentionWrapper] = {}

    def _get_wrapper(
        self,
        num_reqs: int,
        tokens_per_req: int = 1,
    ) -> BatchMLAPagedAttentionWrapper:
        key = (num_reqs, tokens_per_req)
        if key not in self._wrappers:
            logger.info(
                "Creating FlashInfer MLA wrapper for batch=%d tpr=%d",
                num_reqs,
                tokens_per_req,
            )
            self._wrappers[key] = _create_wrapper(
                self._shared_workspace,
                self._device,
                num_reqs,
                self._max_pages_per_req,
                tokens_per_req,
            )
        return self._wrappers[key]

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> FlashInferMLADecodeMetadata:
        del max_seq_len, query_start_loc_cpu, query_start_loc_device

        num_reqs = seq_lens_device.shape[0]
        page_size = self._page_size
        device = seq_lens_device.device

        num_pages_per_req = (seq_lens_device + page_size - 1) // page_size
        kv_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        torch.cumsum(num_pages_per_req, dim=0, out=kv_indptr[1:])
        block_idx = torch.arange(block_table_tensor.shape[1], device=device)
        mask = block_idx.unsqueeze(0) < num_pages_per_req.unsqueeze(1)
        kv_indices = block_table_tensor[mask].to(torch.int32)

        tokens_per_req = num_decode_tokens // num_reqs if num_reqs > 0 else 1
        qo_indptr = torch.arange(
            0,
            num_reqs * tokens_per_req + 1,
            tokens_per_req,
            dtype=torch.int32,
            device=device,
        )

        wrapper = self._get_wrapper(num_reqs, tokens_per_req)
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            seq_lens_device.to(torch.int32),
            num_heads=self.num_heads * self.dcp_world_size,
            head_dim_ckv=self._kv_lora_rank,
            head_dim_kpe=self._qk_rope_head_dim,
            page_size=page_size,
            causal=False,
            sm_scale=(
                1.0 / (self._qk_nope_head_dim + self._qk_rope_head_dim) ** 0.5
            ),
            q_data_type=self.model_config.dtype,
            kv_data_type=self.model_config.dtype,
        )

        return FlashInferMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            wrapper=wrapper,
        )


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        return None


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):
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
                "FlashInferMLAImpl does not support alibi_slopes, "
                "sliding_window, or logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Only decoder attention is supported")

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert attn_metadata.decode.wrapper is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
        else:
            q_nope = q[:, :, : self.kv_lora_rank]
            q_pe = q[:, :, self.kv_lora_rank :]

        ckv_cache = kv_c_and_k_pe_cache[:, :, : self.kv_lora_rank].unsqueeze(2)
        kpe_cache = kv_c_and_k_pe_cache[:, :, self.kv_lora_rank :].unsqueeze(2)

        wrapper = attn_metadata.decode.wrapper
        output, lse = wrapper.run(
            q_nope.contiguous(),
            q_pe.contiguous(),
            ckv_cache,
            kpe_cache,
            return_lse=True,
            return_lse_base_on_e=True,
        )

        v_scale = layer._v_scale_float
        if v_scale != 1.0:
            output = output * v_scale

        return output.contiguous(), lse
