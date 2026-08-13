# SPDX-License-Identifier: Apache-2.0
"""Environment-gated tensors for offline distribution-fidelity evaluation."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


def _is_capture_rank() -> bool:
    # The LM-head output and final transformer state are replicated across the
    # tensor-parallel group. Its first rank is the single persistent writer.
    return get_tp_group().is_first_rank


def _request_directory(root: str, req_id: str) -> Path:
    safe_req_id = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in req_id
    )
    request_dir = Path(root) / safe_req_id
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def _save_tensor(
    tensor: torch.Tensor,
    *,
    output_path: Path,
    key: str,
    metadata: dict[str, str],
) -> None:
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite distribution capture: {output_path}")
    from safetensors.torch import save_file

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file({key: tensor}, str(temporary_path), metadata=metadata)
    os.replace(temporary_path, output_path)


def capture_pre_lm_head_prompt_hidden_states(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    input_batch: Any,
    prompt_lens: np.ndarray,
) -> None:
    """Store final-normalized prompt rows without evaluating the LM head."""
    capture_root = os.environ.get("VLLM_KLD_HIDDEN_CAPTURE_DIR")
    if not capture_root or not _is_capture_rank():
        return

    normalize = getattr(model, "compute_pre_lm_head_hidden_states", None)
    if normalize is None:
        raise RuntimeError(
            "VLLM_KLD_HIDDEN_CAPTURE_DIR requires a model that exposes "
            "compute_pre_lm_head_hidden_states"
        )

    for batch_index, req_id in enumerate(input_batch.req_ids):
        if not input_batch.is_prefilling_np[batch_index]:
            continue
        state_index = int(input_batch.idx_mapping_np[batch_index])
        row_start = int(input_batch.num_computed_prefill_tokens_np[batch_index])
        prompt_rows = int(prompt_lens[state_index])
        scheduled_rows = int(input_batch.num_scheduled_tokens[batch_index])
        row_end = min(row_start + scheduled_rows, prompt_rows)
        if row_start >= row_end:
            continue

        query_offset = int(input_batch.query_start_loc_np[batch_index])
        row_count = row_end - row_start
        request_hidden_states = hidden_states[query_offset : query_offset + row_count]
        normalized = normalize(request_hidden_states)
        if normalized.dtype != torch.bfloat16:
            raise RuntimeError(
                "Pre-LM-head capture requires native BF16 states; got "
                f"{normalized.dtype} for request {req_id}"
            )

        output_path = _request_directory(capture_root, req_id) / (
            f"hidden.rows-{row_start:06d}-{row_end:06d}.safetensors"
        )
        normalized_cpu = normalized.detach().to(device="cpu").contiguous()
        _save_tensor(
            normalized_cpu,
            output_path=output_path,
            key="hidden_states",
            metadata={
                "request_id": req_id,
                "row_start": str(row_start),
                "row_end": str(row_end),
                "semantic_point": "after_final_rmsnorm_before_lm_head",
            },
        )
        logger.info(
            "Saved pre-LM-head rows [%d, %d) shape=%s to %s",
            row_start,
            row_end,
            tuple(normalized_cpu.shape),
            output_path,
        )


def live_prompt_logit_capture_enabled() -> bool:
    """Return whether this rank is the designated live-logit writer."""
    return bool(os.environ.get("VLLM_KLD_LIVE_LOGIT_CAPTURE_DIR")) and (
        _is_capture_rank()
    )


def capture_live_prompt_logits(
    logits: torch.Tensor,
    *,
    req_id: str,
    row_start: int,
) -> None:
    """Store native LM-head output before log-softmax or sampling."""
    capture_root = os.environ.get("VLLM_KLD_LIVE_LOGIT_CAPTURE_DIR")
    if not capture_root or not _is_capture_rank():
        return
    if logits.dtype != torch.bfloat16:
        raise RuntimeError(
            "Live KLD capture requires native BF16 logits; got "
            f"{logits.dtype} for request {req_id}"
        )

    row_end = row_start + logits.shape[0]
    output_path = _request_directory(capture_root, req_id) / (
        f"logits.rows-{row_start:06d}-{row_end:06d}.safetensors"
    )
    logits_cpu = logits.detach().to(device="cpu").contiguous()
    _save_tensor(
        logits_cpu,
        output_path=output_path,
        key="logits",
        metadata={
            "request_id": req_id,
            "row_start": str(row_start),
            "row_end": str(row_end),
            "semantic_point": "live_lm_head_output_before_sampling",
            "vocab_size": str(logits.shape[-1]),
        },
    )
    logger.info(
        "Saved live prompt-logit rows [%d, %d) shape=%s to %s",
        row_start,
        row_end,
        tuple(logits_cpu.shape),
        output_path,
    )
