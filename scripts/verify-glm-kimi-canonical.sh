#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

glm_mla="$root/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
mla_indexer="$root/vllm/v1/attention/backends/mla/indexer.py"
attn_backend="$root/vllm/v1/attention/backend.py"
attn_utils="$root/vllm/v1/attention/backends/utils.py"
fusion_matcher="$root/vllm/compilation/passes/fusion/matcher_utils.py"
ar_fusion="$root/vllm/compilation/passes/fusion/allreduce_rms_fusion.py"
rejection_sampler="$root/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"
rejection_sampler_utils="$root/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
forward_context="$root/vllm/forward_context.py"
attn_registry="$root/vllm/v1/attention/backends/registry.py"
speculative_config="$root/vllm/config/speculative.py"
spec_dflash="$root/vllm/v1/spec_decode/dflash.py"
llm_base_proposer="$root/vllm/v1/spec_decode/llm_base_proposer.py"
gpu_model_runner="$root/vllm/v1/worker/gpu_model_runner.py"
gpu_input_batch="$root/vllm/v1/worker/gpu_input_batch.py"
ubatch_utils="$root/vllm/v1/worker/ubatch_utils.py"
kimi_run="$root/scripts/run-kimi26-vllm"
glm_run="$root/scripts/run-glm51-vllm"

if [[ ! -f "$kimi_run" && -f /usr/local/bin/run-kimi26-vllm ]]; then
  kimi_run=/usr/local/bin/run-kimi26-vllm
fi
if [[ ! -f "$glm_run" && -f /usr/local/bin/run-glm51-vllm ]]; then
  glm_run=/usr/local/bin/run-glm51-vllm
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

require_marker() {
  local path="$1"
  local marker="$2"
  if ! grep -Fq "$marker" "$path"; then
    echo "missing marker in $path: $marker" >&2
    exit 1
  fi
}

reject_marker() {
  local path="$1"
  local marker="$2"
  if grep -Fq "$marker" "$path"; then
    echo "forbidden marker in $path: $marker" >&2
    exit 1
  fi
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch for $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
}

require_file "$glm_mla"
require_file "$mla_indexer"
require_file "$attn_backend"
require_file "$attn_utils"
require_file "$fusion_matcher"
require_file "$ar_fusion"
require_file "$rejection_sampler"
require_file "$rejection_sampler_utils"
require_file "$forward_context"
require_file "$attn_registry"
require_file "$speculative_config"
require_file "$spec_dflash"
require_file "$llm_base_proposer"
require_file "$gpu_model_runner"
require_file "$gpu_input_batch"
require_file "$ubatch_utils"
require_file "$kimi_run"
require_file "$glm_run"

# GLM DCP long-context coherence fix from the verified 2026-05-08 image.
require_marker "$glm_mla" "_b12x_sparse_mla_signature_flags"
require_marker "$glm_mla" "_sparse_mla_decode_forward_with_lse_vllm_metadata"
require_marker "$glm_mla" "forced_sparse_mla_split_decode_config_for_width"
require_marker "$glm_mla" "nsa_cu_seqlens_k"
require_marker "$glm_mla" "can_return_lse_for_decode: bool = True"
require_marker "$mla_indexer" "class DeepseekV4IndexerBackend"
require_marker "$mla_indexer" "DEEPSEEK_V4_INDEXER"
require_marker "$fusion_matcher" "class MatcherFusedAddRMSNorm"
require_marker "$fusion_matcher" "ir.ops.fused_add_rms_norm"
require_marker "$ar_fusion" "VllmPatternReplacement"
require_marker "$ar_fusion" "CustomAllreduce"
require_marker "$ar_fusion" "rocm_aiter_ops"
require_marker "$rejection_sampler" 'self.rejection_sample_method == "synthetic"'
require_marker "$rejection_sampler" "synthetic_acceptance_rates"
require_marker "$rejection_sampler_utils" "SYNTHETIC_MODE=synthetic_conditional_rates is not None"
require_marker "$rejection_sampler_utils" "def rejection_sample"
require_marker "$forward_context" "def static_forward_context"
require_marker "$forward_context" "BOB_DISABLE_STATIC_HOIST"
require_marker "$gpu_input_batch" "is_spec_decode"

# CPU/GPU sync avoidance from glm51-b12x-a16-padfix-cpuhangfix-20260511.
# This carries an exact CPU-side upper-bound shadow for seq_lens so DCP/B12X
# metadata does not fall back to D2H seq_lens reads in async/spec paths.
require_marker "$attn_backend" "seq_lens_cpu_upper_bound"
require_marker "$attn_utils" "seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound"
require_marker "$spec_dflash" "cad.seq_lens_cpu_upper_bound + num_query_per_req"
require_marker "$llm_base_proposer" "or seq_lens_cpu_upper_bound to avoid D2H sync"
require_marker "$gpu_model_runner" "seq_lens_cpu_upper_bound = seq_lens_cpu"
require_marker "$ubatch_utils" "seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound"
require_marker "$mla_indexer" "seq_lens_cpu_upper_bound"

# Kimi K2.6 launch defaults from the verified 2026-05-10 image.
require_marker "$kimi_run" 'MODEL="${MODEL:-moonshotai/Kimi-K2.6}"'
require_marker "$kimi_run" 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_MLA}"'
require_marker "$kimi_run" 'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"'
require_marker "$kimi_run" "lightseekorg/kimi-k2.6-eagle3-mla"

# GLM launch defaults.
require_marker "$glm_run" 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"'
require_marker "$glm_run" "lukealonso/GLM-5.1-NVFP4-MTP"
require_marker "$glm_run" '"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"'

echo "GLM/Kimi canonical source checks passed."
