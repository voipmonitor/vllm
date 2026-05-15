# GLM/Kimi Canonical Editable Image

This records the first working canonical editable vLLM image validated on
2026-05-12.

## Validated Image

```text
voipmonitor/vllm:glm51-canonical-editable-vllm1e118a8-b12x9436cb8-20260512
```

Validated source state:

```text
vLLM branch: codex/glm51-kimi-canonical-a16-upstream-20260511
vLLM commit: 1e118a830 Fix KV cache profiling config compatibility
B12X commit: 9436cb8 runtime: add GLM B12X A16 stability fixes
Base image: voipmonitor/vllm:glm51-canonical-githead-vllmdb22839-b12x9436cb8-20260512
```

The image is intentionally editable-installed from `/opt/vllm`:

```text
PYTHONPATH=/opt/vllm
vllm.__file__=/opt/vllm/vllm/__init__.py
vllm._C=/opt/vllm/vllm/_C.abi3.so
```

Keep `PYTHONPATH=/opt/vllm`; the base image still contains a
`site-packages/vllm` tree, and without `PYTHONPATH` Python can import the
wrong copy.

## Rebuild

Run from this repository, with a clean git tree:

```bash
MAX_JOBS=128 \
CMAKE_BUILD_PARALLEL_LEVEL=128 \
TORCH_CUDA_ARCH_LIST=12.0a \
scripts/build-glm51-canonical-editable-image.sh
```

The script copies `git archive HEAD` into `/opt/vllm`, runs:

```bash
/opt/venv/bin/python -m pip install -e . --no-deps --no-build-isolation -v
```

and commits a Docker image with:

```text
ENV PYTHONPATH=/opt/vllm
ENTRYPOINT ["/bin/bash"]
CMD ["-lc","sleep infinity"]
```

No extra Python dependencies are installed by this script. It depends on the
base image already containing the known-good CUDA 13, FlashInfer, communicator,
launcher, and B12X runtime stack.

Model-specific runtime settings must be supplied by the recipe. The image should
not carry GLM or Kimi defaults in baked env, because those values are easy to
reuse accidentally across models. The build script clears the model/profile envs
when committing the editable image.

## GLM DCP1 MTP Smoke Run

The validated runtime launches the GLM script explicitly and sets the GLM env
explicitly:

```bash
docker run -d \
  --name glm51-dcp1-mtp-canonical-editable-b12x-a16off \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PORT=5264 \
  -e DCP_SIZE=1 \
  -e TP_SIZE=8 \
  -e GLM51_DISABLE_MTP=0 \
  -e ATTENTION_BACKEND=B12X_MLA_SPARSE \
  -e MOE_BACKEND=b12x \
  -e B12X_MOE_FORCE_A16=0 \
  -e KV_CACHE_DTYPE=fp8 \
  -e GPU_MEMORY_UTILIZATION=0.865 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=64 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=256 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e HF_OVERRIDES='{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}' \
  -e SPEC_CONFIG='{"model":"lukealonso/GLM-5.1-NVFP4-MTP","method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"probabilistic","moe_backend":"b12x"}' \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /cache/jit:/cache/jit \
  voipmonitor/vllm:glm51-canonical-editable-vllm1e118a8-b12x9436cb8-20260512 \
  -lc 'exec /opt/vllm/scripts/run-glm51-vllm'
```

Smoke checks:

```bash
curl -sS http://127.0.0.1:5264/v1/models
docker exec glm51-dcp1-mtp-canonical-editable-b12x-a16off \
  bash -lc 'PYTHONPATH=/opt/vllm /opt/venv/bin/python -c "import vllm, vllm._C; print(vllm.__file__); print(vllm._C.__file__)"'
```

Expected startup signs:

```text
RTX6K NCCL residual-add fusion overlay imported.
Enabled custom fusions: act_quant, allreduce_rms
Using 'B12X' NvFp4 MoE backend
Application startup complete.
```

## Kimi K2.6 Smoke Run

Kimi launches the Kimi script explicitly and overrides any GLM-oriented env:

```bash
docker run -d \
  --name kimi-k26-v3-current \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  --shm-size 16g \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e USE_NCCL_XML=0 \
  -e NCCL_GRAPH_FILE= \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-pr2127.so.2.30.3 \
  -e LD_PRELOAD=/opt/libnccl-pr2127.so.2.30.3 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=8 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=0 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e PORT=5002 \
  -e TP_SIZE=8 \
  -e DCP_SIZE=1 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_BATCHED_TOKENS=16384 \
  -e MAX_NUM_SEQS=128 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=512 \
  -e ATTENTION_BACKEND=TRITON_MLA \
  -e KV_CACHE_DTYPE=fp8 \
  -e KIMI_DISABLE_MTP=0 \
  -e KIMI_SPEC_CONFIG='{"model":"lightseekorg/kimi-k2.6-eagle3-mla","method":"eagle3","num_speculative_tokens":3,"draft_kv_cache_dtype":"fp8","rejection_sample_method":"probabilistic"}' \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /cache/jit:/cache/jit \
  voipmonitor/vllm:glm51-canonical-editable-vllm1e118a8-b12x9436cb8-20260512 \
  -lc 'exec /opt/vllm/scripts/run-kimi26-vllm'
```

Expected startup signs:

```text
vLLM is using nccl==2.30.3
model   moonshotai/Kimi-K2.6
lightseekorg/kimi-k2.6-eagle3-mla
Application startup complete.
```
