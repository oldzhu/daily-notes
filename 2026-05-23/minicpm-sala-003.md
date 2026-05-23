Per the repo rule "Proposal first, no direct code changes," here's the plan for re-testing **iter 1** twice before deciding:

## Proposal — NVFP4-FOS iter-1 retest (×2 accuracy + speed)

### Objective
Reproduce iter-1 config exactly (same quant ckpt + same server args) on a clean instance and measure accuracy variance + speed. This tightens the iter-1 mean estimate (currently 75.98 / 70.27, mean ~73, only 2 samples) before deciding to park.

### What "iter 1" means (verified from commit history)
- **Quantize**: `SOAR_QUANT_PROFILE=nvfp4_fos`, `SOAR_NVFP4_FOUR_OVER_SIX=1`, `SOAR_NVFP4_MAX_CALIB_SEQ_LEN` **unset** (defaults to 4096 in `preprocess_model.py`). Calibration inherits the prepare_env defaults: 90 stratified `qa,mcq,cwe`, seed 20260320, `use_prompt_tokens=1`.
- **Server**: Tier1 long-context (chunk=65536, prefill-max-req=4, sched-cons=0.8), `torch-compile-max-bs=24`, flashinfer, KV fp8_e5m2, mixed-chunk, fused_qk_norm_rope, dense.

The current `prepare_env.sh` (commit `a6b34a41a`) overrides these to iter-2 values inside the `nvfp4_fos` branch (lines ~45–58):
```bash
export SOAR_NVFP4_MAX_CALIB_SEQ_LEN="${SOAR_NVFP4_MAX_CALIB_SEQ_LEN:-16384}"
export SOAR_TIER1_LONG_CONTEXT=0
export SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-8}"
```

### Approach (no code change needed — use env overrides)
For both quantize and server, export these BEFORE `source prepare_env.sh`:
```bash
export SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096
export SOAR_TIER1_LONG_CONTEXT=1
export SOAR_TORCH_COMPILE_MAX_BS=24
```
The `${VAR:-default}` pattern in prepare_env preserves caller-provided values, so these stick. The `SOAR_TIER1_LONG_CONTEXT=0` line on line 58 is unconditional, but it sits inside `if [[ "$SOAR_QUANT_PROFILE" == "nvfp4_fos" ]]` — so we either (a) export `SOAR_TIER1_LONG_CONTEXT=1` AFTER sourcing prepare_env (then re-build SGLANG_SERVER_ARGS), or (b) make a one-line patch to prepare_env to respect the caller value.

**Cleaner: tiny prepare_env patch** — change line 58 from:
```bash
export SOAR_TIER1_LONG_CONTEXT=0
```
to:
```bash
export SOAR_TIER1_LONG_CONTEXT="${SOAR_TIER1_LONG_CONTEXT:-0}"
```
This makes iter-2 still the default for `nvfp4_fos`, but allows iter-1 retest by `export SOAR_TIER1_LONG_CONTEXT=1` upstream. No behavior change unless caller overrides.

### Steps (need user approval before running)
1. **Code**: 1-line patch to `prepare_env.sh` line 58. Commit + push to `minicpm-src`.
2. **Resume fcloud**: `start-instance` (after confirming console JWT fresh).
3. **Re-quantize iter-1 ckpt** at `/root/models/MiniCPM-SALA-NVFP4-FOS-iter1` (separate path; iter-2 ckpt currently occupies the FOS path). Env: `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 SOAR_TIER1_LONG_CONTEXT=1 SOAR_TORCH_COMPILE_MAX_BS=24 SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1 SOAR_QUANT_FORCE=1`. Sync exec, ~3 min.
4. **Restart server** with same env (iter-1 server args).
5. **Accuracy ×2**, **speed all variants**.
6. **Pause instance**, document as `CHANGE_0151_phase_b_four_over_six_002.{en,zh}.md` + chat log + new TEST_RESULTS rows (NVFP4-FOS-1c, -1d).

### Expected outcomes (decision criteria)
- If both runs ≥ 77%: iter 1 was real, FOS works at iter-1 config — keep alive, focus on variance reduction.
- If split (one ≥77%, one <77%): variance still kills it, mean ~74 — park unless we add max-tokens cap.
- If both <77%: iter-1's 75.98 was a one-off, park FOS confidently.

### Risk
Low. One-line guarded env default change; no algorithm change. Reproduces commit `829128503` behavior exactly when `SOAR_TIER1_LONG_CONTEXT=1` is set upstream. Ckpt path isolated so iter-2 artifact preserved.

### Cost
~3 min quantize + ~5 min cudagraph + 2×60 min accuracy + ~5 min speed ≈ **2.3 hours fcloud time**.

---

**Approve to proceed?** Specifically:
1. OK with the 1-line `prepare_env.sh` patch (line 58: hard-set → respect-caller default)?
2. OK to resume fcloud and run quantize + 2× accuracy + speed?
3. OK to use a separate ckpt path `/root/models/MiniCPM-SALA-NVFP4-FOS-iter1` (preserves iter-2 ckpt for diff)?
