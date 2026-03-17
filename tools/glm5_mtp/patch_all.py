#!/usr/bin/env python3
"""Apply ALL GLM-5 MTP patches to vLLM dev217+ in one shot."""
import sys, os

V = "/opt/venv/lib/python3.12/site-packages/vllm"
errors = []

def read(path):
    with open(path) as f:
        return f.read()

def write(path, content):
    with open(path, 'w') as f:
        f.write(content)

def patch(name, path, old, new):
    c = read(path)
    if old not in c:
        if new.split('\n')[1] if '\n' in new else new in c:
            print(f"  SKIP {name}: already applied")
            return True
        errors.append(f"{name}: pattern not found in {os.path.basename(path)}")
        print(f"  FAIL {name}: pattern not found")
        # Debug: show first 80 chars of expected pattern
        print(f"       Expected: {repr(old[:120])}")
        return False
    c = c.replace(old, new, 1)
    write(path, c)
    print(f"  OK   {name}")
    return True

print("=" * 60)
print("Patching vLLM for GLM-5 MTP on SM120 (Blackwell)")
print("=" * 60)

# ============================================================
# 1. cuda.py: major == 10 → major >= 10
# ============================================================
p = f"{V}/platforms/cuda.py"
patch("cuda.py: major>=10",
    p,
    "if device_capability.major == 10:",
    "if device_capability.major >= 10:")

# 1b. cuda.py: sparse→dense fallback when no valid backend found
c = read(p)
old_raise = """        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )"""
new_raise = """        if len(valid_backends_priorities) == 0:
            # Fallback: retry without sparse if that was the blocker
            if attn_selector_config.use_sparse:
                import logging
                logging.getLogger('vllm.platforms.cuda').warning(
                    'No sparse MLA backend on this device (cc %s). '
                    'Falling back to dense MLA.',
                    device_capability.as_version_str())
                fb = attn_selector_config._replace(use_sparse=False)
                valid_backends_priorities, fb_reasons = (
                    cls.get_valid_backends(
                        device_capability=device_capability,
                        attn_selector_config=fb,
                        num_heads=num_heads))
                all_invalid_reasons.update(fb_reasons)
            if len(valid_backends_priorities) == 0:
                raise ValueError(
                    f"No valid attention backend found for {cls.device_name} "
                    f"with {config_str}. Reasons: {reasons_str}."
                )"""
if old_raise in c:
    c = c.replace(old_raise, new_raise, 1)
    write(p, c)
    print("  OK   cuda.py: sparse→dense fallback")
elif "Falling back to dense MLA" in c:
    print("  SKIP cuda.py: sparse→dense fallback (already applied)")
else:
    errors.append("cuda.py: sparse→dense fallback pattern not found")
    print("  FAIL cuda.py: sparse→dense fallback")

# ============================================================
# 2. mla.py: add logger + sparse→dense fallback for indexer
# ============================================================
p = f"{V}/model_executor/layers/mla.py"
c = read(p)
if "Falling back to dense attention" not in c:
    # Add logger import
    if "from vllm.logger import init_logger" not in c:
        c = c.replace(
            "import torch\n\nfrom vllm.config",
            "import torch\n\nfrom vllm.logger import init_logger\n\nlogger = init_logger(__name__)\n\nfrom vllm.config")
    # Add fallback after self.prefix = prefix
    marker = "        self.prefix = prefix\n"
    if marker in c:
        ins = ("\n"
            "        # If sparse was requested but backend is dense, disable indexer.\n"
            "        if self.is_sparse and not self.mla_attn.attn_backend.is_sparse():\n"
            "            logger.warning_once(\n"
            '                "Sparse MLA requested but backend %s is not sparse. "\n'
            '                "Falling back to dense attention.",\n'
            "                self.mla_attn.attn_backend.get_name())\n"
            "            self.is_sparse = False\n")
        c = c.replace(marker, marker + ins, 1)
        write(p, c)
        print("  OK   mla.py: sparse→dense indexer fallback")
    else:
        errors.append("mla.py: 'self.prefix = prefix' not found")
        print("  FAIL mla.py: marker not found")
else:
    print("  SKIP mla.py: sparse→dense (already applied)")

# ============================================================
# 3. mla_attention.py: bfloat16 → auto for C++ cache ops
# ============================================================
p = f"{V}/model_executor/layers/attention/mla_attention.py"
c = read(p)
if "kv_cache_dtype in ('bfloat16', 'float16')" not in c:
    old = "        self.kv_cache_dtype = kv_cache_dtype"
    new = ("        # C++ cache ops only accept 'auto' or fp8. Map others.\n"
           "        if kv_cache_dtype in ('bfloat16', 'float16'):\n"
           "            kv_cache_dtype = 'auto'\n"
           "        self.kv_cache_dtype = kv_cache_dtype")
    if old in c:
        c = c.replace(old, new, 1)
        write(p, c)
        print("  OK   mla_attention.py: dtype fix")
    else:
        errors.append("mla_attention.py: pattern not found")
        print("  FAIL mla_attention.py")
else:
    print("  SKIP mla_attention.py: dtype fix (already applied)")

# ============================================================
# 4. backend.py: bfloat16 → auto in do_kv_cache_update
# ============================================================
p = f"{V}/v1/attention/backend.py"
c = read(p)
if "effective_dtype" not in c:
    old = "        ops.concat_and_cache_mla("
    new = ("        # Map non-quantized dtypes to 'auto' for C++ cache ops.\n"
           "        effective_dtype = kv_cache_dtype\n"
           "        if kv_cache_dtype in ('bfloat16', 'float16'):\n"
           "            effective_dtype = 'auto'\n"
           "        ops.concat_and_cache_mla(")
    if old in c:
        c = c.replace(old, new, 1)
        c = c.replace(
            "            kv_cache_dtype=kv_cache_dtype,\n            k_scale=layer._k_scale,",
            "            kv_cache_dtype=effective_dtype,\n            k_scale=layer._k_scale,", 1)
        write(p, c)
        print("  OK   backend.py: dtype fix")
    else:
        errors.append("backend.py: pattern not found")
        print("  FAIL backend.py")
else:
    print("  SKIP backend.py: dtype fix (already applied)")

# ============================================================
# 5. glm4_moe_mtp.py: safe norm (shared_head.norm with residual)
# ============================================================
p = f"{V}/model_executor/models/glm4_moe_mtp.py"
patch("glm4_moe_mtp.py: safe norm",
    p,
    "        hidden_states = residual + hidden_states",
    "        # Apply shared_head norm before returning.\n"
    "        hidden_states, _ = self.shared_head.norm(hidden_states, residual)")

# ============================================================
# 6. glm4_moe_mtp.py: skip double-norm in compute_logits
# ============================================================
patch("glm4_moe_mtp.py: skip double-norm",
    p,
    "            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)",
    "            mtp_layer.shared_head.head, hidden_states")

# ============================================================
# Verification
# ============================================================
print("\n" + "=" * 60)
print("Verification")
print("=" * 60)
checks = [
    (f"{V}/platforms/cuda.py", "major >= 10", "cuda.py: major>=10"),
    (f"{V}/platforms/cuda.py", "Falling back to dense MLA", "cuda.py: sparse fallback"),
    (f"{V}/model_executor/layers/mla.py", "Falling back to dense attention", "mla.py: sparse fallback"),
    (f"{V}/model_executor/layers/mla.py", "logger = init_logger", "mla.py: logger import"),
    (f"{V}/model_executor/layers/attention/mla_attention.py", "kv_cache_dtype in ('bfloat16'", "mla_attention.py: dtype"),
    (f"{V}/v1/attention/backend.py", "effective_dtype", "backend.py: dtype"),
    (f"{V}/model_executor/models/glm4_moe_mtp.py", "shared_head.norm(hidden_states, residual)", "glm4_moe_mtp.py: norm"),
    (f"{V}/model_executor/models/glm4_moe_mtp.py", "mtp_layer.shared_head.head, hidden_states", "glm4_moe_mtp.py: no double-norm"),
]
all_ok = True
for path, needle, desc in checks:
    ok = needle in read(path)
    print(f"  {'OK' if ok else 'FAIL'}: {desc}")
    if not ok:
        all_ok = False

# Quick import test
print("\n" + "=" * 60)
print("Import test")
print("=" * 60)
try:
    # Test that patched modules can be imported without syntax errors
    import importlib, types
    for mod_path in [
        f"{V}/platforms/cuda.py",
        f"{V}/model_executor/layers/mla.py",
        f"{V}/model_executor/layers/attention/mla_attention.py",
        f"{V}/v1/attention/backend.py",
        f"{V}/model_executor/models/glm4_moe_mtp.py",
    ]:
        with open(mod_path) as f:
            compile(f.read(), mod_path, 'exec')
        print(f"  OK   {os.path.basename(mod_path)} compiles")
except SyntaxError as e:
    print(f"  FAIL SyntaxError: {e}")
    all_ok = False

if errors:
    print(f"\nErrors: {errors}")
if all_ok and not errors:
    print("\n*** ALL PATCHES APPLIED SUCCESSFULLY ***")
else:
    print("\n*** SOME PATCHES FAILED — check output above ***")
    sys.exit(1)
