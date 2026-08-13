# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from vllm.v1.worker.gpu import distribution_capture


class FinalNormModel(torch.nn.Module):
    def compute_pre_lm_head_hidden_states(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states.mul(2)


def test_hidden_capture_preserves_prompt_row_alignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setenv("VLLM_KLD_HIDDEN_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(distribution_capture, "_is_capture_rank", lambda: True)
    model = FinalNormModel()

    def make_batch(computed: int, scheduled: int) -> SimpleNamespace:
        return SimpleNamespace(
            idx_mapping_np=np.array([0]),
            is_prefilling_np=np.array([True]),
            num_computed_prefill_tokens_np=np.array([computed]),
            num_scheduled_tokens=np.array([scheduled]),
            query_start_loc_np=np.array([0, scheduled]),
            req_ids=["request/unsafe"],
        )

    first = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    distribution_capture.capture_pre_lm_head_prompt_hidden_states(
        model, first, make_batch(0, 2), np.array([5])
    )
    second = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    distribution_capture.capture_pre_lm_head_prompt_hidden_states(
        model, second, make_batch(2, 3), np.array([5])
    )

    request_dir = tmp_path / "request_unsafe"
    first_path = request_dir / "hidden.rows-000000-000002.safetensors"
    second_path = request_dir / "hidden.rows-000002-000005.safetensors"
    torch.testing.assert_close(load_file(first_path)["hidden_states"], first.mul(2))
    torch.testing.assert_close(load_file(second_path)["hidden_states"], second.mul(2))
