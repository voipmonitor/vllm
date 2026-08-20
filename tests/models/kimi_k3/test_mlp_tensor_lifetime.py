# SPDX-License-Identifier: Apache-2.0

import weakref

import torch
from torch import nn

from vllm.models.kimi_k3.nvidia.model import KimiMLP


class _GateUp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_ref: weakref.ReferenceType[torch.Tensor] | None = None

    def forward(self, x: torch.Tensor):
        output = torch.cat((x, x), dim=-1)
        self.output_ref = weakref.ref(output)
        return output, None


class _Activation(nn.Module):
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        return gate_up.chunk(2, dim=-1)[0].clone()


class _Down(nn.Module):
    def __init__(self, gate_up: _GateUp) -> None:
        super().__init__()
        self.gate_up = gate_up

    def forward(self, x: torch.Tensor):
        assert self.gate_up.output_ref is not None
        assert self.gate_up.output_ref() is None
        return x, None


def test_mlp_releases_gate_up_before_down_projection() -> None:
    mlp = KimiMLP.__new__(KimiMLP)
    nn.Module.__init__(mlp)
    gate_up = _GateUp()
    mlp.gate_up_proj = gate_up
    mlp.act_fn = _Activation()
    mlp.down_proj = _Down(gate_up)
    mlp.shard_sequence_parallel = False

    input_tensor = torch.randn(2, 4)
    torch.testing.assert_close(mlp(input_tensor), input_tensor)
