#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

glm_mla="$root/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
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
require_file "$kimi_run"
require_file "$glm_run"

# GLM DCP long-context coherence fix from the verified 2026-05-08 image.
require_marker "$glm_mla" "_b12x_split_decode_final_lse_kernel"
require_marker "$glm_mla" "_sparse_mla_split_decode_forward_with_lse_vllm_metadata"
require_marker "$glm_mla" "decode_inline_lse"
require_marker "$glm_mla" "nsa_cu_seqlens_k"
require_marker "$glm_mla" "can_return_lse_for_decode: bool = True"

# Kimi K2.6 launch defaults from the verified 2026-05-10 image.
require_marker "$kimi_run" 'MODEL="${MODEL:-moonshotai/Kimi-K2.6}"'
require_marker "$kimi_run" 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_MLA}"'
require_marker "$kimi_run" "lightseekorg/kimi-k2.6-eagle3-mla"
require_marker "$kimi_run" '"draft_attention_backend":"TRITON_MLA"'

# GLM launch defaults.
require_marker "$glm_run" 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"'
require_marker "$glm_run" "lukealonso/GLM-5.1-NVFP4-MTP"

require_sha256 "$glm_mla" "bd59de56a2356ea8cf5b44f54e499f19184ea3c97987a06450e399d5ec7ef8c1"
require_sha256 "$kimi_run" "4cc0484a6ab7a886aa9d8b82a761936edf4751f6b3fa323d9f988def3879f389"
require_sha256 "$glm_run" "f63d405ef18a1637bcc4896b7f63c43fa67a2d9c411655549722cd8b32949dcd"

installed_glm_mla="/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
if [[ -f "$installed_glm_mla" ]]; then
  require_sha256 "$installed_glm_mla" "bd59de56a2356ea8cf5b44f54e499f19184ea3c97987a06450e399d5ec7ef8c1"
fi

echo "GLM/Kimi canonical source checks passed."
