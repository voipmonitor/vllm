# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import time

import torch

from vllm.config import SpeculativeConfig
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.probabilistic_rejection_sampler_utils import (
    probabilistic_rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.synthetic_rejection_sampler_utils import (
    compute_synthetic_rejection_sampler_params,
    synthetic_rejection_sample,
)


@triton.jit
def _strict_rejection_sample_kernel(
    sampled_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_stride,
    num_sampled_ptr,  # [num_reqs]
    target_sampled_ptr,  # [num_draft_tokens + num_reqs]
    input_ids_ptr,  # [num_draft_tokens + num_reqs]
    cu_num_logits_ptr,  # [num_reqs + 1]
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            target_sampled = tl.load(target_sampled_ptr + start_idx + i)
            draft_sampled = tl.load(input_ids_ptr + start_idx + i + 1)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_sampled)
            num_sampled += 1
            if target_sampled != draft_sampled:
                rejected = True
    if not rejected:
        target_sampled = tl.load(target_sampled_ptr + start_idx + num_tokens - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_tokens - 1, target_sampled
        )
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


def strict_rejection_sample(
    # [num_draft_tokens + num_reqs]
    target_sampled: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    num_speculative_steps,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = target_sampled.new_empty(num_reqs, num_speculative_steps + 1)
    num_sampled = target_sampled.new_empty(num_reqs, dtype=torch.int32)
    _strict_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_sampled,
        draft_sampled,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.rejection_sample_method = spec_config.rejection_sample_method
        if self.rejection_sample_method == "synthetic":
            synthetic_acceptance_rate = spec_config.synthetic_acceptance_rate
            if (
                synthetic_acceptance_rate is None
                or not 0.0 <= synthetic_acceptance_rate <= 1.0
            ):
                raise ValueError(
                    f"synthetic_acceptance_rate must be in [0, 1], "
                    f"but got {synthetic_acceptance_rate}"
                )
            self.base_acceptance_rate, self.decay_factor = (
                compute_synthetic_rejection_sampler_params(
                    synthetic_acceptance_rate, self.num_speculative_steps
                )
            )
        self._diag_enabled = os.environ.get("VLLM_MTP_PROB_DIAG", "0") == "1"
        self._diag_log = os.environ.get("VLLM_MTP_PROB_DIAG_LOG",
                                        "/diag/mtp_prob_diag.jsonl")
        self._diag_max_calls = int(os.environ.get("VLLM_MTP_PROB_DIAG_MAX_CALLS",
                                                  "20"))
        self._diag_calls = 0

    def _maybe_log_prob_diag(
        self,
        input_batch: InputBatch,
        processed_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_sampled: torch.Tensor,
    ) -> None:
        if not self._diag_enabled or self._diag_calls >= self._diag_max_calls:
            return
        self._diag_calls += 1
        try:
            local_pos = input_batch.expanded_local_pos
            req_idx = input_batch.expanded_idx_mapping
            row = torch.arange(local_pos.numel(), device=local_pos.device)
            next_row = row + 1
            mask = (
                (local_pos < self.num_speculative_steps)
                & (next_row < draft_sampled.numel())
                & (req_idx == req_idx[next_row.clamp(max=req_idx.numel() - 1)])
            )
            row = row[mask]
            if row.numel() == 0:
                return
            next_row = row + 1
            step = local_pos[row].to(torch.long)
            req = req_idx[row].to(torch.long)
            draft_ids = draft_sampled[next_row].to(torch.long)

            target_lp = processed_logits[row].log_softmax(dim=-1)
            target_draft_lp = target_lp.gather(1, draft_ids[:, None]).squeeze(1)
            draft_lp = draft_logits[req, step].log_softmax(dim=-1)
            draft_draft_lp = draft_lp.gather(1, draft_ids[:, None]).squeeze(1)
            expected_accept = torch.minimum(
                torch.ones_like(target_draft_lp),
                torch.exp(target_draft_lp - draft_draft_lp),
            )

            by_step = {}
            for pos in range(self.num_speculative_steps):
                pos_mask = step == pos
                if not bool(pos_mask.any()):
                    continue
                by_step[str(pos)] = {
                    "n": int(pos_mask.sum().item()),
                    "target_lp_mean": float(target_draft_lp[pos_mask].mean().item()),
                    "draft_lp_mean": float(draft_draft_lp[pos_mask].mean().item()),
                    "logp_delta_mean": float(
                        (target_draft_lp[pos_mask] - draft_draft_lp[pos_mask])
                        .mean()
                        .item()
                    ),
                    "expected_accept_mean": float(
                        expected_accept[pos_mask].mean().item()
                    ),
                }

            rec = {
                "time": time.time(),
                "call": self._diag_calls,
                "num_reqs": int(input_batch.cu_num_logits.shape[0] - 1),
                "num_logits": int(processed_logits.shape[0]),
                "rows": int(row.numel()),
                "by_step": by_step,
            }
            with open(self._diag_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        except Exception as e:
            rec = {
                "time": time.time(),
                "call": self._diag_calls,
                "error": repr(e),
            }
            with open(self._diag_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = input_batch.cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return compute_topk_logprobs(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        if self.rejection_sample_method == "strict":
            sampler_output = self.sampler(logits, input_batch)
            logprobs_tensors = sampler_output.logprobs_tensors
            sampled, num_sampled = strict_rejection_sample(
                sampler_output.sampled_token_ids.view(-1),
                draft_sampled,
                input_batch.cu_num_logits,
                self.num_speculative_steps,
            )
        elif self.rejection_sample_method == "probabilistic":
            assert draft_logits is not None
            pos = input_batch.positions[input_batch.logits_indices]
            processed_logits = self.sampler.apply_sampling_params(
                logits,
                input_batch.expanded_idx_mapping,
                input_batch.idx_mapping_np,
                pos,
                draft_sampled,
                input_batch.expanded_local_pos,
            )
            self._maybe_log_prob_diag(
                input_batch,
                processed_logits,
                draft_logits,
                draft_sampled,
            )
            sampled, num_sampled = probabilistic_rejection_sample(
                processed_logits,
                draft_logits,
                draft_sampled,
                input_batch.cu_num_logits,
                pos,
                input_batch.idx_mapping,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                self.num_speculative_steps,
            )
            logprobs_tensors = self._get_logprobs_tensors(
                input_batch,
                sampled,
                num_sampled,
                processed_logits
                if self.sampler.logprobs_mode == "processed_logprobs"
                else logits,
            )
        elif self.rejection_sample_method == "synthetic":
            sampler_output = self.sampler(logits, input_batch)
            logprobs_tensors = sampler_output.logprobs_tensors
            sampled, num_sampled = synthetic_rejection_sample(
                sampler_output.sampled_token_ids.view(-1),
                draft_sampled,
                input_batch.cu_num_logits,
                input_batch.positions[input_batch.logits_indices],
                input_batch.idx_mapping,
                self.sampler.sampling_states.seeds.gpu,
                self.base_acceptance_rate,
                self.decay_factor,
                self.num_speculative_steps,
            )
        else:
            raise ValueError(
                f"Unknown rejection sample method: {self.rejection_sample_method}"
            )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
