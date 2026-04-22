# Kimi 2.5/2.6 MLA + DCP + SpecDecode Tracking

This document is a virtual tracking issue for splitting the current working Docker state into upstream-sized PRs, then recomposing them to verify parity.

## Base
- Docker code base commit: `47fcb8ca68c1027ba7eb7a9056bb4596ee284221`
- Fork: `voipmonitor/vllm`
- Target upstream repo: `vllm-project/vllm`

## Upstream-candidate PR buckets
1. MLA + DCP + FP8 KV support
2. Async speculative proposer race fix
3. Draft-specific attention backend + draft KV dtype
4. Qwen3 / Qwen3-DFlash local argmax support
5. Llama Eagle3 local argmax support

## Existing upstream context
- `#34795` MLA + DCP + FP8 KV support
- `#39419` local argmax reduction for large-vocab spec decode
- `#39930` DFlash backend selection
- `#39995` DFlash + FlashInfer experiments

## Local-only / not for upstream in current form
- `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN` env workaround
- XML / NCCL_GRAPH_FILE scoping workarounds
- experimental Triton MLA split/reduction tuning
- local NCCL 2.29.7 patched build and no-XML validation
- PCIe custom allreduce opt-in env hack

## Verification plan
1. Create topic branches for each PR bucket.
2. Reapply only the relevant hunks from the working Docker tree.
3. Create an integration branch by merging/cherry-picking those branches.
4. Diff the integration branch against the working Docker tree.
5. If parity is reached, open upstream issue/PRs plus a final Kimi recipe issue.
