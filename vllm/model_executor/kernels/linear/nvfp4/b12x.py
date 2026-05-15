# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

import torch

import vllm.envs as envs
from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.platforms import current_platform

_B12X_PAD_M_TO_POW2 = envs.VLLM_B12X_PAD_M_TO_POW2


def _import_b12x():
    try:
        return importlib.import_module("b12x")
    except ImportError:
        return None


def _import_b12x_fp4():
    try:
        return importlib.import_module("b12x.cute.fp4")
    except ImportError:
        return None


def _import_b12x_dense():
    try:
        return importlib.import_module("b12x.gemm.dense")
    except ImportError:
        return None


def _next_pow2_ge(n: int) -> int:
    """Smallest power of 2 >= n; n must be >= 1."""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


@torch.library.custom_op(
    "vllm_b12x::nvfp4_dense_gemm", mutates_args=(), device_types="cuda"
)
def _b12x_nvfp4_dense_gemm(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    output_size: int,
    pad_m_to_pow2: bool,
) -> torch.Tensor:
    orig_m = x_2d.shape[0]
    m = orig_m
    x_2d = x_2d.contiguous()
    if pad_m_to_pow2:
        # b12x dense_gemm currently faults on decode-style M=1 shapes
        # (for example Kimi EAGLE3 q_b_proj: M=1, N=1536, K=1536).
        # Pad all small-M launches to at least 2 rows before the existing
        # power-of-two padding path and slice the extra rows after GEMM.
        padded_m = max(2, _next_pow2_ge(orig_m))
        if padded_m != orig_m:
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, padded_m - orig_m))
            m = padded_m

    fp4 = _import_b12x_fp4()
    dense = _import_b12x_dense()
    if fp4 is None or dense is None:
        raise ImportError("b12x is not installed or importable")

    x_packed, x_scale_swizzled = scaled_fp4_quant(
        x_2d,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="cutlass",
    )
    x_scale = fp4.as_grouped_scale_view(
        x_scale_swizzled.view(torch.uint8).unsqueeze(0),
        m,
        x_2d.shape[1],
    )

    out = torch.empty((m, output_size, 1), device=x_2d.device, dtype=x_2d.dtype)
    dense.dense_gemm(
        (x_packed.unsqueeze(-1), x_scale),
        (weight.unsqueeze(-1), weight_scale.unsqueeze(-1)),
        out=out,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=str(x_2d.dtype).split(".")[-1],
        sf_vec_size=16,
    )
    return out[:orig_m, :, 0]


@_b12x_nvfp4_dense_gemm.register_fake
def _(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    output_size: int,
    pad_m_to_pow2: bool,
) -> torch.Tensor:
    del weight, weight_scale, input_global_scale_inv, alpha, pad_m_to_pow2
    return x_2d.new_empty((x_2d.shape[0], output_size))


class B12xNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the optional external b12x SM12x backend."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x NVFP4 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x NVFP4 kernels require a Blackwell 12x device"
        if _import_b12x() is None:
            return False, "b12x Python package is not importable"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        del config
        return True, None

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        fp4 = _import_b12x_fp4()
        dense = _import_b12x_dense()
        if fp4 is None or dense is None:
            raise ImportError("b12x is not installed or importable")
        self._as_grouped_scale_view = fp4.as_grouped_scale_view
        self._swizzle_block_scale = fp4.swizzle_block_scale
        self._dense_gemm = dense.dense_gemm

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = torch.nn.Parameter(
            self._swizzle_block_scale(layer.weight_scale.data),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out = _b12x_nvfp4_dense_gemm(
            x_2d,
            layer.weight,
            layer.weight_scale,
            layer.input_global_scale_inv,
            layer.alpha,
            output_size,
            _B12X_PAD_M_TO_POW2,
        )
        assert out.dtype == output_dtype
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
