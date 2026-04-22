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

## Simulated branches in `voipmonitor/vllm`
- Tracking branch / virtual issue:
  - `track/kimi-mla-upstream-split-20260422`
  - draft PR: `https://github.com/voipmonitor/vllm/pull/2`
- Upstream-candidate topic branches:
  - `sim/pr-34795-mla-dcp-fp8`
  - `sim/pr-async-spec-proposer-race`
  - `sim/pr-draft-backend-kv-dtype`
  - `sim/pr-qwen3-local-argmax`
  - `sim/pr-llama-eagle3-local-argmax`
- Local-only runtime branches:
  - `sim/local-spec-disable-threshold`
  - `sim/local-pcie-custom-allreduce`
  - `sim/local-triton-mla-tuning`
- Integration branch:
  - `sim/integration-kimi-mla-20260422`

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

## Integration status
`sim/integration-kimi-mla-20260422` already recomposes these commits:
- `[Core] enable FP8 KV cache with DCP for MLA`
- `[SpecDecode] fix async proposer synchronization`
- `[SpecDecode] allow draft-specific attention backend and KV dtype`
- `[SpecDecode] add local argmax helpers for Qwen3 drafts`
- `[SpecDecode] add local argmax helper for Llama Eagle3`
- `[Local] add seq-length gate for speculative decode`
- `[Local] add PCIe custom allreduce opt-in`
- `[Local] carry current Triton MLA tuning`

Current parity check against live docker tree (`/opt/vllm`) is exact for:
- `tests/distributed/test_context_parallel.py`
- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/cp_utils.py`
- `vllm/model_executor/models/qwen3.py`
- `vllm/model_executor/models/qwen3_dflash.py`
- `vllm/model_executor/models/llama_eagle3.py`
- `vllm/v1/core/sched/async_scheduler.py`
- `vllm/distributed/device_communicators/custom_all_reduce.py`
- `vllm/v1/attention/backends/mla/triton_mla.py`
- `vllm/v1/attention/ops/triton_decode_attention.py`

Still unmatched vs live tree:
- `vllm/config/speculative.py`
  - only because live tree also contains local DFlash window-size additions
- `vllm/v1/spec_decode/eagle.py`
  - only because live tree also contains a local warning/fallback tweak for draft vocab remapping
- `vllm/v1/attention/backends/flash_attn.py`
  - comment/whitespace-only difference

Out-of-scope live-only files still not represented as simulated PRs:
- `vllm/config/attention.py`
- `vllm/v1/attention/selector.py`
- `vllm/v1/spec_decode/dflash.py`
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/v1/attention/backends/mla/flashinfer_mla.py`
- `vllm/distributed/device_communicators/cuda_communicator.py`
- build/requirements noise and backup files

## Benchmarks
### DCP4 with XML-scoped fix
Reference benchmark already measured earlier on the functional XML-based setup:
- summary JSON: `/tmp/llm_decode_bench_dcp4.json`
- selected decode throughput:
  - `ctx=0, C=1`: `85.5 tok/s`
  - `ctx=16k, C=1`: `52.7 tok/s`
  - `ctx=32k, C=1`: `52.7 tok/s`
  - `ctx=64k, C=1`: `52.7 tok/s`
- prefill:
  - `8k`: `7885 tok/s`
  - `16k`: `7998 tok/s`
  - `32k`: `7693 tok/s`

### DCP4 without XML, using patched NCCL 2.29.7
This run used patched NCCL from `voipmonitor/nccl` and **no** `NCCL_GRAPH_FILE`:
- summary JSON: `/tmp/llm_decode_bench_dcp4_noxml_patchednccl.json`
- selected decode throughput:
  - `ctx=0, C=1`: `76.4 tok/s`
  - `ctx=16k, C=1`: `56.7 tok/s`
  - `ctx=32k, C=1`: `56.7 tok/s`
  - `ctx=64k, C=1`: `56.7 tok/s`
- prefill:
  - `8k`: `2663 tok/s`
  - `16k`: `795 tok/s`
  - `32k`: `1414 tok/s`

### Current interpretation
- Patched NCCL **does** work end-to-end without `NCCL_GRAPH_FILE`.
- But the no-XML path is **not** yet performance-equivalent to the XML-based setup.
- Decode at long context is roughly comparable or slightly better without XML.
- Short-context decode is worse without XML.
- Prefill is dramatically worse without XML.

Conclusion for now:
- we cannot drop the XML workaround yet if we want current best end-to-end performance
- therefore the old XML-scoping hack should not be upstreamed, but it also cannot be deleted from the working runtime recipe until no-XML performance is recovered
