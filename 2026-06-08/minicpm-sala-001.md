C and let us temporarily hold on accuracy and move to Step 2 (3–4 days): true W4A8 (INT4 storage + FP8 MMA), and 1st let us verify below in fcloud:
The W4A8 attempt (CHANGE_W4A8_001, commit 7ce21c3f5) preserved accuracy (79.20%, C=1.0) but slowed S1 from 121.71s→265s because the implementation up-casted INT4 weights to FP8 storage at load time, doubling weight bandwidth and killing decode bs=1.  and then go to he correct implementation — INT4 stored, dequant-to-FP8 fused into the GEMM kernel — is filed as PROPOSAL_W4A8_REAL_001 and was never built. This is the single highest-impact item on the catalog. Estimated upside on the official long-context Smax tier (GEMM-bound): −20% to −35%.
========

Understood. Plan:
1. **Disable mcq cap default** (option C) — revert to v22/v24 baseline behavior. Commit + push.
2. **Locate W4A8 docs** to ground the verification proposal.
3. **Propose fcloud verification plan** for the old W4A8 (commit `7ce21c3f5`) to reproduce the S1 regression before designing the real fused-dequant kernel. 

Replacing 9 lines with 16 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)


```

```
Now locate the W4A8 docs to ground the verification proposal:  

Searched for files matching `**/docs/soar_2026_changes/*W4A8*`, 12 matches

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline 7ce21c3f5 -1 2>&1 ; git log --oneline --all | grep -iE "w4a8|7ce21c" | head -10

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4A8_REAL_001.en.md#1-1), lines 1 to 120

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_002.en.md#1-1), lines 1 to 80

Committing the mcq-cap revert:  

Ran terminal command:  cd /home/oldzhu/sglang && git add benchmark/soar/demo_sala/prepare_env.sh && git commit -m "v24-iter1 revert: disable SOAR_MCQ_MAX_TOKENS_CAP default (was 4096)

Round-1 test @ cap=4096 (commit ef2d27d84) regressed mcq from 14/30 to
10/30 — cap truncated legitimate 4k-16k deep-think samples. Overall
accuracy fell to 75.38% (normalized ~93.4% → C=0 risk). Speed gain
+12.4% (2724s vs 3108s) confirmed mechanism works, but net Score
impact is negative.

Default flipped back to 0 (disabled, byte-equivalent to v22/v24).
The serving_chat.py clamp code path remains in place and can be
re-enabled with a larger value (e.g. SOAR_MCQ_MAX_TOKENS_CAP=16384)
for future testing." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Revert pushed as `8ce03caaf`. Now the W4A8 verification proposal:

---

## Step 2 — Proposal: Verify old W4A8 (`SOAR_W4A8_FP8_GEMM=1`) on fcloud

### Background grounding (from PROPOSAL_W4A8_REAL_001.en.md + CHANGE_W4A8_001_iteration_002.en.md)

- **Old path (commit `7ce21c3f5`)**: at GPTQ load time, dequantize INT4 → BF16 → re-quantize to **FP8 storage**, then cutlass FP8 blockwise GEMM. This was **mislabeled W4A8 — it is actually W8A8 FP8 blockwise**. 4-bit weight bandwidth advantage thrown away at load time.
- **Recorded regression** (2026-04 fcloud reading): S1 **+118%** (121.71→265.32s), S8 **+56%**, Smax **+30%**. Accuracy preserved (79.20%, C=1.0).
- **Code still in tree**: env-gated by `SOAR_W4A8_FP8_GEMM` (default 0) in prepare_env.sh line ~306, plumbed through gptq.py and minicpm.py.
- **Real W4A8** target: keep INT4 storage end-to-end + use SM120 FP8 QMMA (296 TF) tensor cores via mixed-input GEMM (Machete-style or custom CUTLASS 3.x).

### Why re-verify the old path before building the new one?

1. **Confirm the regression magnitude on current HEAD**: a lot has changed since 2026-04 (Tier1 long-context defaults, fused-qk-norm-rope, torch_compile_max_bs=24, flashinfer backend default, etc.). The reported 265s S1 was measured on the `chunk=32K / prefill-max-req=1 / sched-cons=1.0 / torch-compile-max-bs=8` family — different from current v24 baseline (`chunk=65K / prefill-max-req=4 / sched-cons=0.8 / torch-compile-max-bs=24`).
2. **Quantify the exact bandwidth penalty to motivate the new design**: if S1 ratio holds at ≥2x, that's a strong empirical anchor for "INT4 storage is mandatory" in the real-W4A8 proposal.
3. **Sanity-check that the env gate still works**: ensures the `SOAR_W4A8_FP8_GEMM=1` plumbing is alive on current code path. If broken, fix before building real W4A8 on top of it.
4. **Free baseline data point** for the Optimization Catalog (currently the catalog cites the old 265s number; updating it with current-config measurement avoids being misled later).

### Test plan (one fcloud round, no source edits)

| Step | Command | Notes |
|---|---|---|
| 1 | `start-instance` | console JWT fresh (3.75 days left) |
| 2 | `sync` | pulls `8ce03caaf` (mcq cap disabled, byte-equivalent to v24 server args) |
| 3 | Override env on fcloud: `export SOAR_W4A8_FP8_GEMM=1` *inside* the restart-server script | `restart-server` sources prepare_env.sh in a fresh shell; we need to set the override **before** sourcing |
| 4 | `restart-server` (with override) → `wait-server` | Expect ~3-5 min start (longer due to FP8 conversion at load) |
| 5 | Quick smoke (Paris completion) | confirm server alive |
| 6 | `speed --variant s1` | the most affected tier; primary signal |
| 7 | `speed --variant s8` | secondary |
| 8 | `speed --variant smax` | tertiary |
| 9 | (skip accuracy — already verified neutral in iter-002; saves ~50 min) | |
| 10 | `pause-instance` | |

**Implementation of step 3**: the cleanest way is a small one-off script on fcloud that does `SOAR_W4A8_FP8_GEMM=1 source ./prepare_env.sh && python -m sglang.launch_server ...`. Alternatively I can teach `fcloud_workflow.py restart-server` to accept `--extra-env KEY=VALUE`. Recommend: **one-off override via direct fcloud exec** (no python_workflow.py changes for a single test).

### Expected outcomes / decision tree

| Result on S1 | Interpretation | Next action |
|---|---|---|
| ≥ 1.5x v24-S1 (≥ ~165s) | Regression mostly reproduces; current config doesn't help | Proceed to PROPOSAL_W4A8_REAL_001 implementation |
| 1.0–1.5x v24-S1 | Improvement from torch.compile-max-bs=24 / Tier1 args + FP8 path | Investigate which config flag closed the gap; possibly cheaper than building new kernel |
| < 1.0x v24-S1 (faster!) | FP8 blockwise GEMM now beats Marlin on current config | Major surprise — would be the cheapest W4A8-equivalent win available; reproducibility-check before celebrating |

### Estimated time
- Setup + start + wait: ~10 min
- 3 speed runs: ~30 min total (S1 longer due to regression)
- Total fcloud time: ~45 min

### Risk
- Minimal. Read-only re-test; no source edits beyond the already-committed mcq revert. Existing W4A8 plumbing has been in the tree for 3 weeks and survived multiple fcloud rounds.

### What this does NOT cover
- The real W4A8 (INT4-storage + fused FP8-dequant kernel) — that's a separate engineering effort (3-4 days per PROPOSAL_W4A8_REAL_001). This step is just **verification of the prior regression** before committing to that build.

---

**Confirm to proceed?** Two yes/no questions:

1. ✅ / ❌ Run the W4A8 verification on fcloud now (start-instance → set `SOAR_W4A8_FP8_GEMM=1` → restart → speed s1/s8/smax → pause)?
2. ✅ / ❌ Skip accuracy for this run (it was already verified neutral in iter-002 — saves ~50 min)?  



