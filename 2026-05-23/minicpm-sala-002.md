Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#2080-2080), lines 2080 to 2130

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1364-1364), lines 1364 to 1410

Now let me check `load_calibration_texts` to understand how SAMPLING/TASK_INCLUDE/USE_PROMPT_TOKENS env vars flow in:         

Searched for regex `SOAR_GPTQ_CALIBRATION_SAMPLING|SOAR_GPTQ_CALIBRATION_TASK_INCLUDE|SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS|def load_calibration_texts` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 6 results

Confirmed. Here's the breakdown:

## Calibration env vars — apply to NVFP4 too?

**Yes, all of them apply.** The "SOAR_GPTQ_CALIBRATION_*" naming is historical — these vars actually control the shared `load_calibration_texts(...)` helper, which is called by **both** GPTQ and NVFP4 paths (`run_nvfp4_quantization` calls it at line ~1394).

So the active NVFP4 calibration is:

| Env var | Effective value | Effect on NVFP4 |
|---|---|---|
| `SOAR_GPTQ_CALIBRATION_FILE` | `perf_public_set.jsonl` | Source dataset ✓ |
| `SOAR_GPTQ_CALIBRATION_SAMPLES` | **90** | 90 samples (not 32 — I was wrong earlier; the default in prepare_env.sh overrides argparse default) |
| `SOAR_GPTQ_CALIBRATION_SAMPLING` | `stratified` | Stratified across tasks ✓ |
| `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` | `qa,mcq,cwe` | Filters dataset to these 3 tasks (excludes `niah`, `fwe`) ✓ |
| `SOAR_GPTQ_CALIBRATION_TASK_BALANCE` | `1` | Balances across the 3 included tasks ✓ |
| `SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS` | `1` | Uses `prompt_tokens` field instead of plain text — **this is the long-context content** ✓ |
| `SOAR_GPTQ_CALIBRATION_SEED` | `20260320` | RNG seed ✓ |
| `SOAR_NVFP4_MAX_CALIB_SEQ_LEN` | `4096` | **NVFP4-specific cap** — long prompts get truncated to 4096 tokens during calibration |

So **correction to my earlier answer**: we are using **90 stratified samples from qa/mcq/cwe with full prompt tokens** — not 32 short questions. Calibration data quality is actually quite good.

**The one NVFP4-specific knob to be aware of**: `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096`. Even though we feed `prompt_tokens` (which on `cwe`/`qa` can be 60k–95k tokens), modelopt only sees the first 4096 of each. For activation `_amax` calibration on long-context attention, this likely **misses the true activation distribution** of long-context inference. Bumping this to 16384 or 32768 is a cheap, plausible accuracy lever.

## `SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8`

- **Weights only**, not KV cache. KV cache dtype is set separately by `KV_CACHE_DTYPE_ARG` (currently `fp8_e5m2`).
- This preset tells the **GPTQ** path: "make Q/K/V projections w8 (8-bit), other layers w4 (4-bit)" — it's GPTQ's heterogeneous bitwidth schedule.
- **Has zero effect on the NVFP4 path.** `run_nvfp4_quantization` does not consult `SOAR_GPTQ_MIXED_PRECISION_PRESET` at all — modelopt's NVFP4 quantizer applies uniform 4-bit-with-fp8-scale to every Linear in the include list, regardless of this preset.

So when running `SOAR_QUANT_PROFILE=nvfp4_fos`, the `sparse_qkv_w8` preset is dead config — NVFP4 quantizes Q/K/V at fp4 just like every other Linear. **This is potentially a real source of accuracy loss**: the GPTQ baseline keeps Q/K/V at w8 because they're sensitive; NVFP4-FOS does not have an equivalent "spare Q/K/V" mechanism wired in.

## Summary of the two real levers

1. **`SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` is short** for a long-context model. Bumping to 16384–32768 is cheap and likely helps.
2. **No QKV-w8 protection** in NVFP4 path. The champion config likely either uses more calibration samples, longer calibration sequences, or has a layer skip-list for sensitive projections (none of which we have today).

Both fit well with the next iteration:
- Bump `SOAR_NVFP4_MAX_CALIB_SEQ_LEN` to 16384.
- Switch to conservative scheduling profile (your earlier concern from the variance result).
- Re-quantize → re-test accuracy 2× for variance check.

Want me to draft this iteration as a proposal for your review?
