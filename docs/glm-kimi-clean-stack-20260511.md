# GLM/Kimi Clean Stack Reconstruction

This document is the reconstruction contract for the GLM-5.1 and Kimi-K2.6
RTX6000/SM120 vLLM branch.

The goal is a clean vLLM branch based on upstream `main`, with our runtime
changes applied as readable topic commits. Docker images are evidence for
what worked, not source trees to overlay wholesale.

## Clean Branch

Base branch:

```text
upstream/main
```

Current reconstruction branch:

```text
codex/glm51-kimi-clean-stack-20260511
```

Local worktree:

```text
/root/vllm-clean-stack-20260511
```

Do not use the previous canonical branch as the base. It contains mixed
full-file overlays from multiple upstream eras.

## Reference Images

These images are the frozen runtime references:

| Purpose | Image | Created |
| --- | --- | --- |
| Kimi known-good runtime | `voipmonitor/vllm:glm51-kimi-20260510` | 2026-05-10 16:26 UTC |
| GLM/Kimi communicator and DCP baseline | `voipmonitor/vllm:glm51-kimi-comm-20260508` | 2026-05-08 04:38 UTC |
| GLM B12X A16 plus CPU/GPU hang fix | `voipmonitor/vllm:glm51-b12x-a16-padfix-cpuhangfix-20260511` | 2026-05-11 17:18 UTC |
| Contaminated reference only | `voipmonitor/vllm:glm51-kimi-canonical-fi-git-cutedsl45-quickcopy-20260511` | 2026-05-11 21:42 UTC |

Snapshot extracts are stored outside git:

```text
/root/vllm-snapshots/20260511
```

The `canonical-*` image is not a trusted base. It is useful only to find
recent edits that may not exist in the older verified images.

## Historical GLM Upstream Tracker

The original upstream tracker is:

```text
https://github.com/vllm-project/vllm/issues/37113
```

It defines the first GLM-5.1 NVFP4 DCP/MTP stack and links these PRs:

| PR | Topic | Status for this reconstruction |
| --- | --- | --- |
| `#39550` | GLM appended NextN/MTP draft layer loading | Port as model-loading topic |
| `#39632` | MTP runtime metadata reuse and DCP-aware draft plumbing | Port only relevant still-needed hunks |
| `#39633` | PCIe custom allreduce eligibility | Superseded by newer communicator work, but keep as historical root |
| `#39634` | Import optional external `b12x` into vLLM NVFP4 dense/MoE selection | Port as standalone B12X integration topic |
| `#39635` | FlashInfer MLA DCP/MTP decode path | Historical root for MLA/DCP correctness, superseded by newer B12X sparse MLA work |

## Historical Kimi Upstream Tracker

The Kimi reconstruction tracker is:

```text
https://github.com/vllm-project/vllm/issues/40608
```

It separates the old validated Docker snapshot from the clean upstream-main
reconstruction path. The important upstream PR roots are:

| PR | Topic | Status for this reconstruction |
| --- | --- | --- |
| `#40609` | MLA plus DCP plus FP8 KV cache support | Ported as `attention: support MLA DCP with fp8 KV cache` |
| `#40610` | Async proposer synchronization | Ported as `spec-decode: synchronize async proposer work` |
| `#40611` | Draft-specific attention backend and draft KV dtype | Ported as `spec-decode: isolate draft backend and cache config` |
| `#40654` | Avoid `seq_lens_cpu` GPU-to-CPU sync | Already present in current `upstream/main`; do not duplicate older local experiments |
| `#40750` | Consolidated TRITON_MLA full CUDA graph Kimi MTP stack | Ported as `attention: enable TRITON_MLA full graph decode tuning` |
| `#40895` | Export parallel/DCP config in metrics | Ported as `metrics: export parallel config info` |

Superseded Kimi items:

- `#40612` is Llama-Eagle3 local-argmax specific, not part of the Kimi-K2.6
  MLA draft path unless a later GLM/Kimi test proves otherwise.
- `#40613` is a draft seq-length gate discussion, superseded by the final
  Kimi MTP path.
- `#40614` is superseded by `#40750`.

The tracker recipe used draft model `lightseekorg/kimi-k2.5-eagle3-mla` in the
first upstream issue text. Our current runtime contract must use
`lightseekorg/kimi-k2.6-eagle3-mla`.

## Patch Series Order

Use this order so each commit is reviewable and rebaseable.

1. `glm-nextn-mtp-loading`

   Port the GLM appended NextN/MTP layer loading support. Historical root:
   upstream PR `#39550`.

2. `spec-decode-core-runtime`

   Port speculative decode runtime changes that are still required by both
   Kimi Eagle/MTP and GLM MTP. This includes DCP-aware draft metadata reuse and
   async proposer synchronization.

   Historical roots: upstream PRs `#39632` and `#40610`.

3. `kimi-draft-backend-kv`

   Port Kimi target/draft split support:

   - draft-specific attention backend
   - draft KV cache dtype
   - draft CP override handling

   Historical root: upstream PR `#40611`.

4. `kimi-triton-mla-dcp-fp8`

   Port Kimi `TRITON_MLA` support for:

   - MLA plus DCP plus FP8 KV cache
   - full CUDA graph decode for uniform spec-verify shapes
   - DCP correctness for spec-verify decode
   - batch-aware KV split selection
   - SM120/FP8 tuning table

   Historical roots: upstream PRs `#40609` and `#40750`.

5. `pcie-communicator-selector`

   Port PCIe custom allreduce work as explicit communicator commits:

   - `VLLM_ENABLE_PCIE_ALLREDUCE`
   - `VLLM_PCIE_ALLREDUCE_BACKEND`
   - C++ PCIe allreduce backend
   - row-aware cutoff controls such as `VLLM_CPP_AR_1STAGE_NCCL_CUTOFF`
   - `VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS`
   - opt-in RTX6K fused allreduce experiments

   Historical root: upstream PR `#39633`; current runtime evidence comes from
   `glm51-kimi-20260510` and `glm51-kimi-comm-20260508`.

6. `b12x-import-into-vllm`

   Port the optional external `b12x` integration into vLLM. This is separate
   from sparse MLA and separate from A16.

   Expected areas:

   - `vllm/config/kernel.py`
   - `vllm/envs.py`
   - `vllm/model_executor/kernels/linear/nvfp4/b12x.py`
   - `vllm/model_executor/layers/fused_moe/...`
   - `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
   - NVFP4 weight preparation and backend registration

   Historical root: upstream PR `#39634`.

7. `glm-b12x-sparse-mla-dcp-lse`

   Port GLM B12X sparse MLA DCP correctness:

   - B12X sparse MLA backend registration
   - DCP local sequence length handling
   - LSE returned to vLLM for DCP output combine
   - split-path forcing when LSE is required
   - inline LSE/decode path fixes
   - long-context DCP2/DCP4/DCP8 coherence

   Runtime evidence: `glm51-kimi-comm-20260508` and later GLM DCP tests.

8. `glm-sparse-mla-fanout`

   Port the historical sparse MLA pre-attention/mid-attention fan-out patch:

   - `VLLM_ENABLE_MLA_PREATTN_FANOUT`
   - `BOB_ENABLE_EAGLE3_FANOUT`
   - `execute_in_parallel` tracing fallback and callable aux-stream support
   - optional `kw`/`q_raw` inputs to the GLM sparse indexer

   This is a performance path for GLM sparse MLA long prefill. It auto-enables
   only for validated sparse MLA backends such as `B12X_MLA_SPARSE` and
   `FLASHMLA_SPARSE`; Kimi `TRITON_MLA` remains unaffected unless explicitly
   forced.

   Runtime evidence: all inspected Docker snapshots carried this patch:
   `glm51-kimi-comm-20260508`, `glm51-kimi-20260510`, and
   `glm51-b12x-a16-padfix-cpuhangfix-20260511`.

9. `cpu-sync-cpuhangfix`

   Preserve and verify the `seq_lens_cpu_upper_bound` fix from upstream
   `#40654` and from `glm51-b12x-a16-padfix-cpuhangfix-20260511`.

   Files observed in the runtime diff from `padfix` to `cpuhangfix`:

   - `vllm/v1/attention/backend.py`
   - `vllm/v1/attention/backends/utils.py`
   - `vllm/v1/attention/backends/mla/indexer.py`
   - `vllm/v1/spec_decode/dflash.py`
   - `vllm/v1/spec_decode/llm_base_proposer.py`
   - `vllm/v1/worker/gpu_model_runner.py`
   - `vllm/v1/worker/ubatch_utils.py`

   Purpose: avoid D2H `seq_lens` reads or hangs in async/spec/DCP metadata
   paths by carrying a CPU-side upper-bound shadow. Current `upstream/main`
   already contains the upstream form of this fix, so this topic is primarily
   a verifier/audit step unless the GLM B12X path needs additional propagation.

10. `b12x-moe-a16`

   Port `B12X_MOE_FORCE_A16=1` support and padding fixes without falling back
   to FlashInfer/CUTLASS for the target GLM MoE path.

   Runtime evidence: `glm51-b12x-a16-padfix-cpuhangfix-20260511`.

11. `kimi-launch-contract`

   Preserve the Kimi-K2.6 runtime contract:

   - model `moonshotai/Kimi-K2.6`
   - draft `lightseekorg/kimi-k2.6-eagle3-mla`
   - attention backend `TRITON_MLA`
   - no GLM B12X sparse indexer defaults leaking into Kimi

   Runtime evidence: `glm51-kimi-20260510`.

12. `glm-launch-contract`

   Preserve the GLM-5.1 runtime contract:

   - model `lukealonso/GLM-5.1-NVFP4-MTP`
   - attention backend `B12X_MLA_SPARSE`
   - correct `index_topk_pattern`
   - target `MOE_BACKEND=b12x`
   - draft MoE remains `flashinfer_cutlass` unless explicitly changed

13. `metrics-and-observability`

    Ported from `#40895`. This is not a speed path, but it prevents
    KV-budget/DCP confusion in future benchmark tooling.

14. `build-runtime`

    Keep Docker/build changes as the last topic:

    - patched NCCL PR `#2127`
    - FlashInfer git/nightly build decision
    - CUTE DSL version
    - external `b12x` version/source
    - final Docker labels that identify exact git commits

## External B12X Branch

The vLLM branch is not self-contained. The latest verified GLM A16 runtime also
depends on external `b12x` changes.

Canonical external B12X branch:

```text
https://github.com/local-inference-lab/b12x/tree/codex/glm51-kimi-b12x-a16-cpuhangfix-20260511
```

Base:

```text
local-inference-lab/b12x@3917cb2fe5a2118eaab8b68f7710c71aad9e4b1c
```

Current head:

```text
9f6a0248497d9bd920338a3c1a20be9f62d1ffe7
```

Topic commits:

| Commit | Topic |
| --- | --- |
| `65a44db` | Keep sparse MLA split selection stable when vLLM/DCP requires LSE |
| `474fc60` | Reuse NSA indexer TMA descriptors from the B12X workspace instead of rebuilding them on the hot path |
| `5578ec5` | Make TP MoE cache keys safe under `torch.inference_mode()` and disable eager exact dynamic workspace by default |
| `9f6a024` | Restore the PCIe oneshot completion barrier used by the verified Docker images |

This branch matches the `site-b12x` tree from
`voipmonitor/vllm:glm51-b12x-a16-padfix-cpuhangfix-20260511` after excluding
`__pycache__` and `.pyc` files.

Do not build the final common Docker from plain upstream `b12x@3917cb2`; it
misses the DCP/LSE behavior, NSA indexer CPU-sync reduction, A16 MoE stability
fixes, and the PCIe oneshot completion barrier.

CPU/GPU sync audit notes for this branch:

- Sparse MLA width shrink still has an `.item()` fallback only when the call is
  not graph-stable and LSE is not required. The vLLM DCP/LSE path keeps static
  width and does not use that sync.
- NSA page-id validation has `.item()` reads only behind
  `B12X_NSA_VALIDATE_PAGE_IDS=1`, which is off by default.
- NSA tiled top-k `torch.cuda.synchronize()` is in explicit prewarm code, not in
  serving hot path.
- TP MoE exact dynamic workspace sizing remains opt-in through
  `B12X_MOE_EAGER_EXACT_DYNAMIC=1`; default runtime avoids that routing-derived
  D2H sizing path.

## GLM rtx6kpro Audit

The following GLM documentation was audited against this branch and the external
B12X branch:

```text
https://github.com/local-inference-lab/rtx6kpro/tree/master/models/glm5.1
https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.md
```

Findings that are already captured by this clean stack:

- `B12X_MLA_SPARSE` is the required GLM attention path. The dense MLA
  comparison from `2026-04-20` is quality evidence against replacing NSA/sparse
  MLA with dense MLA as a workaround.
- GLM DCP correctness depends on returning valid LSE from the B12X sparse MLA
  split path. This is covered by the vLLM sparse MLA integration plus the
  external B12X LSE/split commits.
- GLM long-prefill performance depends on the sparse MLA fan-out patch that was
  present in all inspected verified Docker images. The current clean stack
  carries it as an explicit source patch instead of relying on an image overlay.
- The historical vanilla `b12x==0.11.1` overlay had three important runtime
  deltas: NSA indexer CPU-sync reduction, tiled top-k dynamic layout keys, and
  the PCIe oneshot completion barrier. The current external B12X branch carries
  the equivalent behavior on top of newer B12X.
- Patched NCCL PR `#2127` is a Docker/build contract, not a vLLM source patch.
  The final image must expose `/opt/libnccl-pr2127.so.2.30.3`, set
  `VLLM_NCCL_SO_PATH` or `LD_PRELOAD`, use `NCCL_PROTO=LL,LL128,Simple`, and
  unset `NCCL_GRAPH_FILE` for the no-XML path.
- GLM MTP uses `use_local_argmax_reduction=true`; this is present in the
  current speculative config and proposer code.
- The old `VLLM_MTP_RETURN_NORMALIZED_HIDDEN=1` launcher env has become
  redundant in this branch: `DeepSeekMultiTokenPredictorLayer.forward()` now
  returns `self.shared_head(hidden_states)` unconditionally, and logits/top-token
  paths consume the normalized hidden state directly.
- RTX6K fused allreduce/add/RMS remains experimental. The documented GLM result
  says no-end-barrier fused add corrupted GLM MTP hidden states, while the
  end-barrier variant was correct but slower at high concurrency. The safe
  default remains `VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0`.
- For DCP greater than 1, GLM docs consistently show
  `VLLM_ENABLE_PCIE_ALLREDUCE=0` as the default unless a new per-shape selector
  proves otherwise. DCP1 can use the C++ PCIe allreduce selector.
- The correct GLM `index_topk_pattern` has no extra trailing `F`:
  `FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS`.

Items that still need an explicit decision before final release:

- `VLLM_SPEC_ACCEPT_THRESHOLD_ACC` and `VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE` are
  documented in the older GLM pages, but the current branch does not implement
  those env variables. The old backport used them as a throughput/quality
  experiment in `vllm/v1/sample/rejection_sampler.py`. Do not keep exporting
  these envs as if they work; either remove them from launchers or re-port them
  intentionally as an opt-in non-distribution-preserving mode.
- `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` appears in the 2026-05-02 GLM vLLM
  command. The env still exists in vLLM, but later GLM docs do not consistently
  require it. Treat it as an A/B launch knob, not as an assumed source patch.
- The 2026-05-02 report says B12X MoE and attention/indexer paths were prewarmed
  for common M values up to `8192`. The current vLLM branch folds MoE M shapes
  into stable buckets through `VLLM_B12X_MOE_STATIC_CHUNK_TOKENS` and the B12X
  branch has explicit NSA tiled top-k prewarm code, but the final Docker launch
  still needs a startup check that first real long prefill does not trigger CUTE
  recompilation.
- The SGLang-only patches under
  `compare-dense-mla-vs-nsa-benchmark-2026-04-20/patches` are not vLLM source
  patches. They should only be revisited if we build a SGLang image again.

## Hard Rules

- Do not copy entire source files from Docker snapshots into the clean branch.
- Every patch must be attributable to one topic above.
- If a file needs many hunks, split by behavior, not by source file.
- Verifier checks must assert features, not only file hashes.
- Kimi and GLM launch defaults must not leak into each other.
- Current `canonical-*` images are evidence only, not a trusted base.

## Open Decisions

1. Tree attention/spec-decode stack:

   Verified images contain `tree_attn.py`, but current upstream no longer uses
   that path. The clean branch keeps the newer upstream spec-decode path and
   does not backport `TREE_ATTN` unless a concrete runtime test proves it is
   still required.

2. B12X sparse MLA ownership:

   Separate external `b12x` package changes from vLLM integration changes.
   If vLLM requires B12X API behavior, document the exact external b12x commit
   and avoid hiding it in Docker-only state.

3. A16 target path:

   `B12X_MOE_FORCE_A16=1` must stay a B12X target MoE path. Do not solve A16
   failures by silently selecting FlashInfer/CUTLASS.

4. Runtime test matrix:

   Minimum acceptance before release:

   - GLM DCP1/DCP4/DCP8, MTP on/off
   - GLM 70k context coherence
   - GLM A16 target MoE startup and concurrent requests
   - Kimi DCP8 MTP cc1 and cc64 against `glm51-kimi-20260510`
   - communicator stats for PCIe RX/TX and tok/s
