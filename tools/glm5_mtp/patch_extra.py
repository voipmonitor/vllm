#!/usr/bin/env python3
"""Extra patches for dev217+ (beyond patch_all.py)"""
V = "/opt/venv/lib/python3.12/site-packages/vllm"

def read(p):
    with open(p) as f: return f.read()
def write(p, c):
    with open(p, 'w') as f: f.write(c)

# 1. speculative.py: glm_moe_dsa → Glm4MoeMTP BEFORE deepseek fallback
p = f"{V}/config/speculative.py"
c = read(p)
old = '''        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
            hf_config.model_type = "deepseek_mtp"'''
new = '''        initial_architecture = hf_config.architectures[0]
        # GLM-5 MTP: must be checked BEFORE deepseek fallback
        if hf_config.model_type == "glm_moe_dsa":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({"n_predict": n_predict, "architectures": ["Glm4MoeMTPModel"]})
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32"):
            hf_config.model_type = "deepseek_mtp"'''
if old in c:
    c = c.replace(old, new)
    write(p, c)
    print("OK: speculative.py glm_moe_dsa fix")
elif "GLM-5 MTP" in c:
    print("SKIP: speculative.py (already applied)")
else:
    print("FAIL: speculative.py pattern not found")

# 2. flash_attn.py: bfloat16→auto dtype
p = f"{V}/v1/attention/backends/flash_attn.py"
c = read(p)
old = '''        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,'''
new = '''        _kv_dtype = self.kv_cache_dtype
        if _kv_dtype in ('bfloat16', 'float16'):
            _kv_dtype = 'auto'
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            _kv_dtype,'''
if old in c:
    c = c.replace(old, new)
    write(p, c)
    print("OK: flash_attn.py dtype fix")
elif '_kv_dtype' in c:
    print("SKIP: flash_attn.py (already applied)")
else:
    print("FAIL: flash_attn.py pattern not found")

# 3. backend.py: fix ALL concat_and_cache_mla kv_cache_dtype usages
p = f"{V}/v1/attention/backend.py"
c = read(p)
remaining = c.count('kv_cache_dtype=kv_cache_dtype,')
if remaining > 0:
    c = c.replace('kv_cache_dtype=kv_cache_dtype,', 'kv_cache_dtype=effective_dtype,')
    # Ensure effective_dtype defined before second concat_and_cache_mla
    idx1 = c.index('ops.concat_and_cache_mla(')
    idx2 = c.index('ops.concat_and_cache_mla(', idx1+1)
    if 'effective_dtype' not in c[idx2-200:idx2]:
        ins = ("        effective_dtype = kv_cache_dtype\n"
               "        if kv_cache_dtype in ('bfloat16', 'float16'):\n"
               "            effective_dtype = 'auto'\n        ")
        c = c[:idx2] + ins + c[idx2:]
    write(p, c)
    print(f"OK: backend.py fixed {remaining} remaining kv_cache_dtype usages")
else:
    print("SKIP: backend.py (all already fixed)")

# 4. glm4_moe_mtp.py: load_weights guard for missing params (indexer.k_norm etc)
p = f"{V}/model_executor/models/glm4_moe_mtp.py"
with open(p) as f:
    lines = f.readlines()
applied = False
for i, l in enumerate(lines):
    if 'param = params_dict[name]' in l and (i == 0 or 'if name not in params_dict' not in lines[i-1]):
        indent = l[:len(l) - len(l.lstrip())]
        guard = f"{indent}if name not in params_dict:\n{indent}    continue\n"
        lines.insert(i, guard)
        write(p, ''.join(lines))
        print(f"OK: glm4_moe_mtp.py load_weights guard at line {i+1}")
        applied = True
        break
if not applied:
    if any('if name not in params_dict' in l for l in lines):
        print("SKIP: glm4_moe_mtp.py guard (already applied)")
    else:
        print("FAIL: glm4_moe_mtp.py guard pattern not found")

# 5. Clear pycache
import subprocess
subprocess.run(['find', '/opt/venv', '-name', '__pycache__', '-type', 'd', '-exec', 'rm', '-rf', '{}', '+'], capture_output=True)
print("Cleared pycache")
