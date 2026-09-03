# Qwen3.8-Flash-Next TP1 MTP3 decode profiles

Status: **qualified diagnostic capture**.

This directory compares vLLM and SGLang decode execution for
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision
`b797d2e1160b9596b2570e56c1d3590faa09d4ed`. Both servers ran sequentially
on physical GPU 3, an NVIDIA RTX PRO 6000 Blackwell Workstation Edition at
stock clock policy, PCIe Gen5 x16, driver 610.57.04, and a 600 W power limit.

Both captures use TP1, MTP3/NEXTN3, FP8 E4M3 KV cache, language-model-only
mode, one active request, the same 16-token prompt, greedy sampling, and 2,048
requested output tokens. Each raw trace contains exactly eight target-model
markers. Profiling changes absolute latency; use the unprofiled measurements
for throughput and the traces for kernel, launch, synchronization, and overlap
analysis.

## Unprofiled result

One warmup request was excluded. Each reported median covers five sequential
requests. Acceptance varies because the speculative draft sampler does not
honor the request seed deterministically, while verifier throughput remains
stable.

| Engine | Output tok/s | Verifier rounds/s | Accepted tokens/round |
|---|---:|---:|---:|
| vLLM with B12X | 214.405 | 73.318 | 2.913 |
| SGLang | 235.375 | 83.209 | 2.829 |

SGLang is 9.78% faster in median output throughput and 13.49% faster in the
acceptance-independent verifier rate. The verifier-rate difference is the
appropriate optimization target.

## Trace overview

The summary utility counts `execute_context_0(0)_generation_1(4)` as the vLLM
target marker and `step[TARGET_VERIFY bs=1]` as the SGLang target marker.

| Eight-step trace metric, per marker | vLLM with B12X | SGLang |
|---|---:|---:|
| Summed CUDA kernel time | 15.070 ms | 14.841 ms |
| Union CUDA-busy time | 13.347 ms | 12.100 ms |
| CUDA span | 14.666 ms | 12.645 ms |
| Idle time inside CUDA span | 1.319 ms | 0.544 ms |

SGLang reduces the union of GPU work by 9.34%, the CUDA span by 13.78%, and
idle gaps inside that span by 58.73%. Summed kernel time differs by only 1.52%,
which indicates that launch ordering and overlap account for a material part
of the verifier gap.

The top-kernel reports are descriptive rather than direct one-to-one mappings.
The runtimes use different execution paths:

- vLLM uses B12X for QSA, GDN decode, and MXFP8 linear operations and uses
  FlashInfer CUTLASS for NVFP4 MoE.
- SGLang uses its FlashInfer linear-attention path, FlashInfer MXFP8 linear
  kernels, and FlashInfer CUTLASS for NVFP4 MoE.
- vLLM uses FULL_AND_PIECEWISE CUDA graphs. SGLang uses full decode graphs and
  keeps GDN MTP cache mode `none`.

## Files

- `traces/vllm-gpu3-8-target-steps.pt.trace.json.gz`: vLLM rank-0 Torch trace.
- `traces/sglang-gpu3-8-target-steps.trace.json.gz`: SGLang rank-0 Torch trace.
- `results/*-unprofiled-5x2048.json`: per-request timings and verifier counts.
- `results/*-trace-summary.txt`: aggregate timing and top CUDA kernels.
- `measure_engine_decode.py`: common OpenAI-completions measurement client.
- `CONFIGURATION.md`: exact serving and profiler arguments.
- `source/`: exact Python source mounted over the vLLM image during capture.
- `SHA256SUMS`: content hashes for every artifact and source snapshot file.

Open the compressed traces directly in Perfetto or Chrome's trace viewer. The
source snapshot is evidence for the executed Python layer; generated CUDA/CuTe
binaries remain identified by their kernel names in the traces.
