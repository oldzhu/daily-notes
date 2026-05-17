Good news on v22 upload. Answers below.

### Q1 — Will `nvidia-modelopt` break existing deps?

**Risk: low-medium, but real.** Let me show the dependency landscape, then mitigation.

Currently pinned in prepare_env.sh (force-reinstalled, `--no-deps`):

| Package | Version | How pinned |
|---|---|---|
| `torch` | (whatever submission_sim ships, cu128) | not touched by us |
| `transformers` | 4.57.1 | local wheel, `--no-deps` |
| `gptqmodel` | 5.7.0 | local wheel, `--no-deps` |
| `torchao` | 0.9.0 | local wheel, `--no-deps` |
| `flash-attn` | 2.8.3+cu128sm120 | local wheel |
| `huggingface-hub` | 0.34.4 | exact pin |
| `accelerate`, `logbar`, `tokenicer`, `pypcre`, `device-smi`, `threadpoolctl` | latest | `--no-deps` (mostly) |

**`nvidia-modelopt` 0.31.0 declared deps** (from PyPI metadata): `cppimport`, `ninja`, `numpy`, `onnx`, `pulp`, `regex`, `safetensors`, `scipy`, `tqdm`, `pydantic`. Plus the `[hf]` extra adds `transformers`, `accelerate`, `peft`, `datasets`, `huggingface-hub`. **None of those declared bounds collide with our pins** — but the `[hf]` extra would *try* to upgrade `transformers` and `huggingface-hub` if we let it.

**Mitigation (this is exactly why I wrote `--no-deps` in the proposal)**:

```bash
# In prepare_env.sh — install AFTER our pinned wheels, with --no-deps
uv pip install --no-deps "nvidia-modelopt==0.31.0"
# Manually add only the missing pieces (none of these conflict):
uv pip install --no-deps "cppimport" "pulp" "onnx" "pydantic"
# scipy / numpy / tqdm / regex / safetensors / ninja already present in the base image
```

So the install ordering is:
1. our pinned `transformers 4.57.1`, `gptqmodel`, `torchao`, `huggingface-hub 0.34.4` (existing — unchanged)
2. **then** `nvidia-modelopt --no-deps` (cannot upgrade anything we already pinned because of `--no-deps`)
3. then explicit `--no-deps` installs of modelopt's missing transitive deps (cppimport, pulp, onnx, pydantic)

**Verification step** after install (added to existing `pinned_dependency` print block in prepare_env.sh):
```python
for name in ["torch","gptqmodel","transformers","torchao","huggingface_hub","nvidia_modelopt"]:
    print(version)
```
If any version drifted from the pin, prepare_env.sh exits non-zero — same pattern as today.

**Worst-case backstop**: if at runtime modelopt still fails to import for any reason (e.g., a hidden CUDA version check), preprocess automatically falls through to the in-tree `nvfp4_quantize.py` (~120 LOC) — same output format, no modelopt dep at all. That's the safety belt.

### Q4 — "On-site quantization" — explanation

This is **the same model** as your current GPTQ flow. Nothing new here — I just want to be explicit about the rule.

**Competition rule** (https://soar.openbmb.cn/competition, "Submission Constraints"):

> Quantized models must be quantized **on-site** (cannot submit pre-quantized weights). Quantization + evaluation time ≤ 5 hours.

What that means concretely, **today**, in our v22 submission:

```
# Inside the official evaluator's container, after extracting your tarball:
1. bash prepare_env.sh        # install deps
2. bash prepare_model.sh \    # ← THIS is the quantization step
     --input  /root/raw-bf16-model \
     --output /root/quantized-model
3. python launch_server.py ... --model-path /root/quantized-model
4. eval_model.py runs accuracy + speed
```

`prepare_model.sh` calls preprocess_model.py which calls our `run_gptq_quantization(...)` — that's the GPTQ INT4 calibration **happening inside the evaluator's container, on the evaluator's GPU, in real time**. We do **not** ship pre-quantized `.safetensors`. Wall-clock budget for the quant step ≈ 90-120 min (well under the 5h total cap).

**Phase A is identical in spirit**:
- ✅ Ship prepare_env.sh + preprocess_model.py + `nvfp4_quantize.py` + raw scripts in tarball
- ✅ The evaluator's container runs `prepare_model.sh` → calls `run_nvfp4_quantization(...)` → produces NVFP4 weights on-site
- ❌ **Do NOT** pre-quantize locally and ship `.safetensors`

**Why the rule exists**: official ranking validates that the quantization is reproducible from the BF16 reference and respects the 5h time budget. Pre-quantized submissions are forbidden so all teams compete on calibration *and* serving cost.

**Why I asked**: I just wanted to confirm we won't try to cheat by shipping pre-baked weights. The answer is: **no, we ship the script, not the weights** — exactly like v22 today.

**Side note on time budget**: NVFP4 calibration via modelopt's `mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG)` is **faster** than GPTQ (no per-block iterative solving — it's per-block round-to-nearest within E2M1 lattice + fp8 scale). Expect 30-60 min vs current GPTQ's ~90 min. The Phase B FourOverSix step adds ~5-10 min on top (one extra MSE comparison per block).

---

### Status

| Q | Decision |
|---|---|
| Q1 | Install modelopt with `--no-deps` + explicit transitive `--no-deps` adds; verify pins post-install; in-tree fallback as backstop. **No conflict expected.** |
| Q2 | Keep `--force-dense-minicpm` ✅ |
| Q3 | Keep KV at FP8_e5m2 ✅ |
| Q4 | On-site quantization (same as today's GPTQ flow) ✅ |

If you confirm Q1's mitigation looks acceptable, I'll proceed to implement Phase A:
1. Edit prepare_env.sh (add `SOAR_QUANT_PROFILE` switch + modelopt install + version verification)
2. Add `run_nvfp4_quantization` to preprocess_model.py + dispatcher
3. Add `nvfp4_quantize.py` fallback
4. Local stub-model sanity (no fcloud yet)
5. Then ask you to start fcloud for the real run

Reply **approve implement** and I'll start. Or push back on any item. 


