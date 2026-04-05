# GLM-5 NVFP4 on Blackwell (SM120) — vLLM 0.19.1

## Prerequisites
pip install b12x  # SM120 MoE kernels (already installed)

## Launch — No MTP
VLLM_LOG_STATS_INTERVAL=1 \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS \
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 NCCL_MIN_NCHANNELS=8 \
SAFETENSORS_FAST_GPU=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model festr2/GLM-5-NVFP4-MTP \
  --host 0.0.0.0 --port 5199 \
  --served-model-name GLM-5 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 40960 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype bfloat16 \
  --enable-prefix-caching --enable-chunked-prefill

## Launch — MTP=1 (recommended)
Add: --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

## Launch — MTP=4 (max throughput, use --gpu-memory-utilization 0.85)
Add: --speculative-config '{"method":"mtp","num_speculative_tokens":4}'

## Expected backends (check in log)
- MoE: B12X (auto-selected on SM120)
- Attention: FLASHINFER_MLA
- FP4 GEMM: FLASHINFER_CUTLASS
- Draft MoE: Unquantized (MTP weights are BF16)

## Benchmark
python3 /mnt/llm_decode_bench.py --port 5199 --skip-prefill --concurrency 1 --contexts 0,16384,32768

## Results (8xRTX PRO 6000 Blackwell, TP=8, C=1)

| Config    | ctx=0  | ctx=16k | ctx=32k |
|-----------|--------|---------|---------|
| No MTP    | 52.7   | 47.7    | 45.7    |
| MTP=1     | 75.5   | 67.5    | 64.7    |
| MTP=2     | 77.6   | 71.7    | 65.6    |
| MTP=4     | 86.5   | 78.5    | 68.6    |

## Git branch
voipmonitor/vllm@glm5-sm120-0.19.1
Tag: glm5-sm120-b12x-working

## Docker
voipmonitor/vllm:cu130-glm5-b12x-mtp

## Mount requirements
docker run -d --name vllm-glm5-cu130 \
  --gpus all --privileged --shm-size 64g \
  -v /mnt:/mnt -v /cache:/cache \
  -v /root/.cache:/root/.cache \
  --entrypoint /bin/bash \
  voipmonitor/vllm:cu130-glm5-b12x-mtp \
  -c "sleep infinity"
