# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional MTP MoE activation amax capture for offline quantization.

This file is intentionally env-gated and has no effect unless
``VLLM_MTP_AMAX_DUMP_PATH`` is set.  It is a calibration aid for mixed-precision
GLM/Kimi-style MTP layers, not an inference optimization.
"""

from __future__ import annotations

import atexit
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation

try:
    from torch._dynamo import disable as torch_dynamo_disable
except ImportError:

    def torch_dynamo_disable(fn):  # type: ignore[no-untyped-def]
        return fn


logger = init_logger(__name__)

DEFAULT_MTP_LAYER_REGEX = (
    r"(^|\.)layers\.78\.mlp$|(^|\.)layers\.78\.mtp_block\.mlp$"
)
NVFP4_ACT_SCALE_DENOM = 448.0


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _rank_id() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    for key in ("RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is None:
        return False
    return bool(is_compiling())


class MTPAmaxCalibrator:
    """Collect per-expert routed-MoE activation maxima.

    Captured tensors:
    - ``w13_input_amax``: input to gate/up projections, per global expert.
    - ``w2_input_amax``: input to down projection after gated activation.

    ``w2_input_amax`` requires floating-point ``w13_weight`` and is skipped for
    already-quantized runtime layers.  Run this against the BF16 MTP checkpoint
    when collecting calibration data.
    """

    def __init__(
        self,
        *,
        prefix: str,
        num_experts: int,
        dump_path: Path,
        dump_interval_s: float,
        collect_w2: bool,
    ) -> None:
        self.prefix = prefix
        self.num_experts = num_experts
        self.dump_path = dump_path
        self.dump_interval_s = dump_interval_s
        self.collect_w2 = collect_w2
        self.rank = _rank_id()

        self.w13_input_amax: torch.Tensor | None = None
        self.w2_input_amax: torch.Tensor | None = None
        self.sample_count: torch.Tensor | None = None
        self._updates = 0
        self._last_dump = time.monotonic()
        self._warned_no_w2 = False
        self._warned_route_failure = False
        self._registered_atexit = False

    @classmethod
    def from_env(cls, prefix: str, num_experts: int) -> "MTPAmaxCalibrator | None":
        dump_path = os.getenv("VLLM_MTP_AMAX_DUMP_PATH")
        if not dump_path:
            return None

        pattern = os.getenv("VLLM_MTP_AMAX_LAYER_REGEX", DEFAULT_MTP_LAYER_REGEX)
        if pattern and re.search(pattern, prefix) is None:
            return None

        interval_s = float(os.getenv("VLLM_MTP_AMAX_DUMP_INTERVAL", "30"))
        collect_w2 = _env_flag("VLLM_MTP_AMAX_COLLECT_W2", True)
        calibrator = cls(
            prefix=prefix,
            num_experts=num_experts,
            dump_path=Path(dump_path),
            dump_interval_s=interval_s,
            collect_w2=collect_w2,
        )
        logger.info(
            "Enabled MTP MoE amax capture for %s: path=%s interval=%ss "
            "collect_w2=%s",
            prefix,
            dump_path,
            interval_s,
            collect_w2,
        )
        return calibrator

    def _ensure_tensors(self, device: torch.device) -> None:
        if self.w13_input_amax is not None:
            return
        self.w13_input_amax = torch.zeros(
            self.num_experts, device=device, dtype=torch.float32
        )
        self.w2_input_amax = torch.zeros(
            self.num_experts, device=device, dtype=torch.float32
        )
        self.sample_count = torch.zeros(
            self.num_experts, device=device, dtype=torch.int64
        )
        if not self._registered_atexit:
            atexit.register(self.dump)
            self._registered_atexit = True

    def _output_file(self) -> Path:
        path = self.dump_path
        if path.suffix == ".safetensors":
            return path.with_name(f"{path.stem}_rank{self.rank}{path.suffix}")
        name = (
            f"mtp_amax_{_sanitize_filename(self.prefix)}_rank"
            f"{self.rank}.safetensors"
        )
        return path / name

    def _local_expert_id(self, layer: torch.nn.Module, expert_id: int) -> int | None:
        expert_map = getattr(layer, "expert_map", None)
        if expert_map is not None:
            local_id = int(expert_map[expert_id].item())
            return local_id if local_id >= 0 else None
        return expert_id

    @torch_dynamo_disable
    @torch.no_grad()
    def record(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> None:
        if hidden_states.numel() == 0:
            return

        self._ensure_tensors(hidden_states.device)
        assert self.w13_input_amax is not None
        assert self.w2_input_amax is not None
        assert self.sample_count is not None

        try:
            topk_weights, topk_ids = layer.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        except Exception:
            if not self._warned_route_failure:
                logger.exception("Failed to route tokens for MTP amax capture")
                self._warned_route_failure = True
            return

        top_k = topk_ids.shape[-1]
        hidden = hidden_states.detach()
        flat_ids = topk_ids.detach().to(torch.long).reshape(-1)
        flat_hidden = (
            hidden.unsqueeze(1)
            .expand(-1, top_k, -1)
            .reshape(-1, hidden.shape[-1])
        )
        if getattr(layer, "apply_router_weight_on_input", False):
            weights = topk_weights.detach().reshape(-1, 1).to(flat_hidden.dtype)
            flat_hidden = flat_hidden * weights

        valid = (flat_ids >= 0) & (flat_ids < self.num_experts)
        if not bool(valid.any()):
            return

        expert_ids = flat_ids[valid].unique()
        w13_weight = getattr(layer, "w13_weight", None)
        can_collect_w2 = (
            self.collect_w2
            and isinstance(w13_weight, torch.Tensor)
            and w13_weight.is_floating_point()
            and w13_weight.dim() == 3
        )
        if self.collect_w2 and not can_collect_w2 and not self._warned_no_w2:
            logger.warning(
                "MTP amax capture cannot derive w2_input_amax for %s because "
                "w13_weight is not a floating-point [E, 2I, H] tensor. "
                "Run calibration on the BF16 MTP checkpoint.",
                self.prefix,
            )
            self._warned_no_w2 = True

        for expert_id_tensor in expert_ids:
            expert_id = int(expert_id_tensor.item())
            mask = flat_ids == expert_id
            expert_hidden = flat_hidden[mask]
            if expert_hidden.numel() == 0:
                continue

            self.w13_input_amax[expert_id] = torch.maximum(
                self.w13_input_amax[expert_id],
                expert_hidden.abs().amax().to(torch.float32),
            )
            self.sample_count[expert_id] += expert_hidden.shape[0]

            if can_collect_w2:
                local_id = self._local_expert_id(layer, expert_id)
                if local_id is None or local_id >= w13_weight.shape[0]:
                    continue
                expert_w13 = w13_weight[local_id]
                gate_up = expert_hidden.to(expert_w13.dtype).matmul(expert_w13.t())
                out_features = (
                    gate_up.shape[-1] // 2
                    if layer.activation.is_gated
                    else gate_up.shape[-1]
                )
                activated = torch.empty(
                    (gate_up.shape[0], out_features),
                    device=gate_up.device,
                    dtype=gate_up.dtype,
                )
                apply_moe_activation(layer.activation, activated, gate_up)
                self.w2_input_amax[expert_id] = torch.maximum(
                    self.w2_input_amax[expert_id],
                    activated.abs().amax().to(torch.float32),
                )

        self._updates += 1
        now = time.monotonic()
        should_dump = (
            self.dump_interval_s <= 0
            or now - self._last_dump >= self.dump_interval_s
        )
        if should_dump:
            self.dump()
            self._last_dump = now

    @torch_dynamo_disable
    @torch.no_grad()
    def dump(self) -> None:
        if self.w13_input_amax is None:
            return
        assert self.w2_input_amax is not None
        assert self.sample_count is not None

        out_file = self._output_file()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        w13 = self.w13_input_amax.detach().to(torch.float32).cpu()
        w2 = self.w2_input_amax.detach().to(torch.float32).cpu()
        counts = self.sample_count.detach().cpu()
        tensors: dict[str, torch.Tensor] = {
            f"{self.prefix}.experts.w13_input_amax": w13,
            f"{self.prefix}.experts.w2_input_amax": w2,
            f"{self.prefix}.experts.sample_count": counts,
        }
        for expert_id in range(self.num_experts):
            tensors[
                f"{self.prefix}.experts.{expert_id}.gate_proj.input_amax"
            ] = w13[expert_id].reshape(()).clone()
            tensors[
                f"{self.prefix}.experts.{expert_id}.up_proj.input_amax"
            ] = w13[expert_id].reshape(()).clone()
            tensors[
                f"{self.prefix}.experts.{expert_id}.down_proj.input_amax"
            ] = w2[expert_id].reshape(()).clone()

        metadata = {
            "prefix": self.prefix,
            "rank": str(self.rank),
            "updates": str(self._updates),
            "collect_w2": str(self.collect_w2),
            "scale_formula": (
                f"input_scale=1/(input_amax*{NVFP4_ACT_SCALE_DENOM})"
            ),
        }
        tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
        from safetensors.torch import save_file

        save_file(tensors, str(tmp_file), metadata=metadata)
        os.replace(tmp_file, out_file)
