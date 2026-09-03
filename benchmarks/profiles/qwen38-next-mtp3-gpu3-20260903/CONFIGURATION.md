# Capture configuration

## Request

```json
{
  "model": "Qwen3.8-Flash-Next",
  "prompt": "Explain how speculative decoding improves language model inference throughput, including its correctness requirement.",
  "max_tokens": 2048,
  "temperature": 0,
  "top_p": 1,
  "top_k": -1,
  "seed": 20260903,
  "ignore_eos": true
}
```

## vLLM

Image ID:
`sha256:00f1e1ab8906401bae0997f76ffd080f540adf39c64d3772caa786f0ca3905c1`

Runtime versions: vLLM
`0.26.1rc0+glm53.flash.nvfp4.luke.clean.r1.vllme75bcfd.b12x58a046f`,
B12X 1.3.0, FlashInfer 0.6.18+cu133, and PyTorch 2.13.0.

The exact mounted Python files are stored under `source/`. Their development
worktrees had vLLM HEAD `df6aaad5dc8276f25a1902eceaa3859c74f5cc34` and B12X
HEAD `aa398de54f2c1b456dac65e3bc2ee3634100c821`; `SHA256SUMS` is authoritative
for the uncommitted file state used by the server.

Serving arguments:

```text
vllm serve MODEL_REVISION
  --served-model-name Qwen3.8-Flash-Next
  --host 0.0.0.0
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --async-scheduling
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --block-size 64
  --load-format instanttensor
  --gpu-memory-utilization 0.96
  --max-model-len 262144
  --max-num-seqs 4
  --max-num-batched-tokens 6019
  --mm-encoder-tp-mode data
  --mm-processor-cache-gb 0
  --language-model-only
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x"}'
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend flashinfer_cutlass
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
```

The physical attention page selected by the hybrid cache planner is 3,008
tokens even though the requested block size is 64. Relevant environment:

```text
VLLM_SSM_CONV_STATE_LAYOUT=DS
VLLM_PLE_CPU_OFFLOAD=1
VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1
B12X_PCIE_ONESHOT_PDL=1
B12X_MHC_PDL=1
VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
```

Profiler arguments:

```text
--profiler-config.profiler=torch
--profiler-config.torch_profiler_with_stack=true
--profiler-config.torch_profiler_record_shapes=false
--profiler-config.torch_profiler_with_memory=false
--profiler-config.torch_profiler_with_flops=false
--profiler-config.torch_profiler_use_gzip=true
--profiler-config.ignore_frontend=true
--profiler-config.delay_iterations=0
--profiler-config.max_iterations=8
--profiler-config.warmup_iterations=0
--profiler-config.active_iterations=9
--profiler-config.wait_iterations=0
```

Profiling was triggered with `POST /start_profile` after decode began. vLLM
stopped automatically after eight worker steps.

## SGLang

Image ID:
`sha256:07bb7e0f354c606cf77dabc49be95aebb8b2b52a5ba2cc91cc4062c1e93693e3`.
The image reports SGLang build commit
`d91c3682b0b429e4c70df63cd57f819588ce29b0`, CUDA 13.0.3, and FlashInfer
0.6.17.

Serving arguments:

```text
sglang serve
  --served-model-name Qwen3.8-Flash-Next
  --model-path MODEL_REVISION
  --trust-remote-code
  --language-model-only
  --ple-offload-embedding
  --linear-attn-prefill-backend flashinfer
  --linear-attn-decode-backend flashinfer
  --max-mamba-cache-size 24
  --mamba-radix-cache-strategy extra_buffer
  --mamba-track-interval 64
  --mamba-ssm-dtype bfloat16
  --gdn-mtp-cache-mode none
  --tp 1
  --quantization modelopt_mixed
  --kv-cache-dtype fp8_e4m3
  --moe-runner-backend flashinfer_cutlass
  --context-length 262144
  --mem-fraction-static 0.98
  --page-size 64
  --chunked-prefill-size 4096
  --max-running-requests 4
  --speculative-algorithm NEXTN
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
```

The server also enabled a 32 GiB write-through hierarchical cache. It is
inactive for the zero-context single-request decode measurement.

Profiling was triggered after decode began:

```json
{
  "num_steps": 8,
  "activities": ["CPU", "GPU"],
  "with_stack": true,
  "record_shapes": false,
  "profile_id": "qwen38-next-mtp3-sglang-gpu3-8steps"
}
```

SGLang stopped automatically after eight target steps.
