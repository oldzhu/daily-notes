You're right — that's a real inconsistency. Let me correct it.

## The fix

MiniCPM-SALA's 32 layers split into:
- **8 standard attention layers** — have a traditional paged KV cache → **subject to FP4 KV ablation**
- **24 lightning (SimpleGLA) layers** — use recurrent state, **no KV cache to quantize** → not part of M2.0

So **M2.0 ablation should run 8 times, not 32** — once per standard attention layer.

Additional implication: the champion's "first/last layers FP8, middle FP4" pattern in the W5 blog post was on a **different architecture** (presumably all-attention or far fewer SimpleGLA layers). Their first/last layer rule does **not** map cleanly onto our 8 sparse standard layers — we need to find which of OUR 8 standard layers are sensitive, and the answer may not be "first and last" in any obvious sense.

I'll need to survey the architecture to confirm which absolute layer indices (0-31) are the standard attention ones. Let me check:     

Searched for regex `standard|sparse_attention|lightning|simple_gla|layer_type|is_lightning` (`**/python/sglang/srt/models/minicpm.py`), 20 results

Searched for regex `mixer_type|layer_types|attention_layers|std_attn|standard_attn|full_attn` (`**/python/sglang/srt/models/minicpm.py`), 16 results

Confirmed: per-layer mixer comes from `config.mixer_types[layer_id]`. The 8 standard layers are determined by the model config — exact indices to be read from `config.json` of the deployed model on fcloud (or HF), but the principle is solid: only `mixer_type == "minicpm4"` (standard attention) layers have a paged KV cache.

Let me update the plan doc with the correction. 

Replacing 19 lines with 41 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PLAN_post_v18_baseline_M2_kv_mixed_20260426_1029.md)


```

```
## Summary of correction

You're absolutely right — the conflict was real. Corrected:

| | Before | After |
|---|---|---|
| Ablation count | 32 layers | **8 layers** (only `mixer_type == "minicpm4"` standard attention) |
| Compute time | ~3h | ~65min (8 × ~8min including server restarts) |
| Bandwidth-saving ceiling | "halves KV bandwidth" | "halves KV bandwidth **of the 8 standard-attention layers only** — lightning SimpleGLA state I/O for the other 24 layers is unaffected" |

**New consideration added** to the risk register: before committing to M2.1, M2.0 must also produce an estimate of **what fraction of total decode KV-bandwidth comes from the 8 standard layers vs the 24 SimpleGLA recurrent-state I/O**. If the 24 lightning layers dominate the KV bandwidth budget (likely possible given the lightning state I/O optimizations already enabled), then even halving the 8 standard layers' KV reads gives a smaller-than-expected gain. This is an additional go/no-go input.

The champion's "first/last FP8, middle FP4" rule **does not transfer directly** — their architecture is presumably all-attention or has many more attention layers. Our 8 standard layers are scattered among 32, and we'll need to discover their absolute indices from `config.mixer_types` at run time.

---

## You approved actions 1 and 2 — executing now

**Action 1**: survey SGLang for upstream NVFP4 KV cache support
**Action 2**: identify v18 commit hash + produce file-classification table for surgical revert

Now starting both. I'll need fcloud to be **off** during the survey (purely local-repo grep work), and we won't need fcloud until the actual revert validation step.

Starting with action 1 (SGLang upstream survey) and action 2 (v18 commit ID + diff).  

Ran terminal command:  cd /home/oldzhu/sglang && git log --all --oneline | grep -iE 'v18|submit_v18|baseline.*v18|v18.*baseline' | head -30

 

Searched for regex `nvfp4|fp4_e2m1|FP4_E2M1|NVFP4|fp4_kv|kv.*fp4|fp4.*kv` (`**/python/sglang/srt/**`), 20 results

**Excellent — major finding for Action 1**: SGLang already has **`fp4_e2m1` KV cache support upstream**, with `MHATokenToKVPoolFP4` for standard (non-MLA) attention. We don't need to write a memory pool from scratch — we can reuse this and just add a per-layer override.

Let me dig deeper.      

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline mixed_minicpm_cudagraph -25

Searched for regex `MHATokenToKVPoolFP4|class.*FP4` (`**/python/sglang/srt/mem_cache/**`), 4 results

Ran terminal command:  cd /home/oldzhu/sglang && ls -la benchmark/soar/demo_sala/minicpm_sala_submit_v*.tar.gz 2>/dev/null; echo "---"; git log --oneline mixed_minicpm_cudagraph -50 | grep -iE 'v18|v19|test|baseline|submission' | head -30

Ran terminal command:  cd /home/oldzhu/sglang && git log --all --pretty=format:'%h %ad %s' --date=format:'%Y-%m-%d %H:%M' --until='2026-04-14 18:00' --since='2026-04-14 10:00' 2>/dev/null | head -20; echo "---v19 boundary (Apr 20 10:01)---"; git log --all --pretty=format:'%h %ad %s' --date=format:'%Y-%m-%d %H:%M' --until='2026-04-20 10:30' --since='2026-04-20 08:00' 2>/dev/null | head -20

Ran terminal command:  cd /home/oldzhu/sglang && echo "=== commits between v18 (a9f4d43cb) and v19 (08fd86023) ==="; git log --pretty=format:'%h %ad %s' --date=format:'%Y-%m-%d %H:%M' a9f4d43cb..08fd86023 2>/dev/null; echo; echo "=== files changed in that range ==="; git diff --name-only a9f4d43cb..08fd86023 | grep -v '^docs/' | grep -v '^benchmark/soar/demo_sala/.*\.md$' | grep -v 'TEST_RESULTS' | head -50

I have what I need for both action items. Let me deliver the findings cleanly.

## Action 1 results — SGLang upstream FP4 KV cache survey

**Finding: SGLang ALREADY supports FP4 KV cache upstream.** No need to write the memory pool from scratch.

| Component | Location | Status |
|---|---|---|
| `MHATokenToKVPoolFP4` (standard attn pool) | memory_pool.py | ✅ Implemented |
| `MLATokenToKVPoolFP4` (MLA pool, not used by us) | memory_pool.py | ✅ Implemented |
| `--kv-cache-dtype fp4_e2m1` server flag | server_args.py | ✅ Available |
| Auto-creation of FP4 pool on `fp4_e2m1` flag | model_runner_kv_cache_mixin.py | ✅ Wired |

**Implication**: M2.0 can use **Option A (real FP4 storage)** — much cleaner than fake-quant. Plan changes to:

- M2.0: simply set `--kv-cache-dtype fp4_e2m1` globally first, run accuracy → if disastrous (≤75% per champion's number), proceed to per-layer ablation
- For the per-layer override, we add a thin hook in the attention layer init that decides per `layer_id` whether to write to FP4 pool or FP8 pool. Both pools coexist.
- M2.1 is largely **eliminated** — the pool already exists. We mostly need a small change to support **per-layer dtype mixing** (today the pool is global).

**Caveat**: SGLang's current `--kv-cache-dtype fp4_e2m1` is global. To support **mixed FP8+FP4 per layer**, we'll need a small modification to the pool-creation site at model_runner_kv_cache_mixin.py so that **two pools coexist** and each attention layer indexes into the correct one. This is small (~50 lines), much smaller than building a pool from scratch.

---

## Action 2 results — v18 commit identification + file classification

**v18 commit**: `a9f4d43cb` (2026-04-14 15:58 — "Test 18 prep: enable torch.compile (max-bs=8) + record Test 17 results")
Tarball was built at 17:34, ~1.5h after this commit.

**v19 commit**: `08fd86023` (2026-04-20 09:01 — "CHANGE_0120: best scheduling config")
Tarball was built at 10:01, ~1h after.

### Files changed v18 → v19 + classification

| File | Commit | Change | Classification | Reason |
|------|--------|--------|----------------|--------|
| prepare_env.sh | 08fd86023 | CHANGE_0120: prefill-max-req=4, sched-cons=0.8, chunk=65K, max-running=24, mixed-chunk | **REVERT** | Officially proven worse; pre-v18 args won S8/Smax |
| minicpm.py | 96a4c6318, f373fbade, 96304f9cd | M1: fold residual_scale/scale_emb/scale_width at load time, env-gated `SGLANG_MINICPM_FOLD_SCALING` | **REVERT** | Prime suspect for v19 hang at sample 138 (persisted even after FLA revert in v19-c-FLA test). Even though env-gated, eliminates risk surface. |
| preprocess_model.py | 30dc39371, 08213639a | CHANGE_0110 dense-attention GPTQ calibration tuning | **REVERT** | Affects model build; if we ever re-quantize, must use v18 calibration script. (Doesn't affect existing fcloud model, but revert for cleanliness.) |
| chunk.py | 23d1c8ecf | CHANGE_0080 SGLANG_FLA_CHUNK_SIZE env var | **KEEP** | v19-c-FLA test proved reverting drops acc 78.04 → 76.29 (-1.75pt). Net beneficial. |
| chunk_delta_h.py | 23d1c8ecf | CHANGE_0080 FLA chunk-delta tweaks | **KEEP** | Same story, paired with chunk.py |
| eval_model_001.py | (need to inspect) | unknown | **TBD — INSPECT URGENTLY** | Per copilot rules this file MUST never be modified. If it differs from v18, that's a violation that must be fixed regardless of revert decision. |
| copilot-instructions.md | various | docs additions | **KEEP** | Pure documentation, no runtime effect |
| fcloud_workflow.py | various | tooling improvements | **KEEP** | Local-only tooling, no submission impact |

### Plus: post-v19 commits (some already reverted things)

Between v19 tarball and current HEAD, these test-iteration commits applied additional changes:
- `5ab5787e8` test30: fp8_e5m2 → fp8_e4m3 swap  
- `fc8029005` test32: revert v19 scheduling to v18 ✅ (already reverts CHANGE_0120)
- `e625363a8` test33: restore v19 scheduling + disable torch.compile
- `9851c0add` test34: restore Test 20 best combo (chunk=32K, prefill-max-req=1, sched-cons=1.0, running=24, mixed-chunk, torch.compile bs=8, fp8_e5m2 KV) — **this is currently active**

So **HEAD's `prepare_env.sh` is already roughly at "Test 20 best combo", not v19's CHANGE_0120**. The remaining REVERT items are really just minicpm.py and `preprocess_model.py`.

### Open question for you

Confirm the eval script — let me check:     

Ran terminal command:  cd /home/oldzhu/sglang && git diff a9f4d43cb..HEAD -- benchmark/soar/demo_sala/eval_model_001.py | head -50

**⚠️ Eval script integrity violation detected** — but it's a contained one. The diff adds two **opt-in** CLI flags (`--task_filter`, `--num_samples_per_task`) for stratified sampling. **When neither flag is used, behavior is byte-identical to upstream.** The control-flow change is wrapped in conditionals on `args.task_filter` / `args.num_samples_per_task`, both default `None`.

However, the copilot-instructions rule is strict: **"Never modify the eval script. Replace ours byte-for-byte with the official version."** The rule was added precisely to prevent this. Two options:

- **Strict compliance**: revert eval script to v18 byte-for-byte; lose the local stratified-sampling helper for ablation runs.
- **Pragmatic carve-out**: keep additive flags (since default=None preserves upstream behavior); add a code comment + bullet in copilot-instructions noting these specific flags are local-only debug helpers and never used during official submission.

This is a **policy decision** — please pick.

---

## Updated proposed surgical revert plan

```bash
# 1. Confirm v18 baseline reference
V18=a9f4d43cb

# 2. Revert source-code regressions (minicpm.py + preprocess_model.py)
git checkout $V18 -- python/sglang/srt/models/minicpm.py
git checkout $V18 -- benchmark/soar/demo_sala/preprocess_model.py

# 3. Decide on eval_model_001.py per your policy choice
# (Option strict)  git checkout $V18 -- benchmark/soar/demo_sala/eval_model_001.py
# (Option pragmatic) leave as-is

# 4. Skip prepare_env.sh — HEAD's "Test 20 best combo" is post-v19 but already 
#    closer to v18's strategy than v19's CHANGE_0120 was. Verify with diff first:
git diff $V18 HEAD -- benchmark/soar/demo_sala/prepare_env.sh
# Then decide whether to leave HEAD's settings or restore exact v18 settings.

# 5. KEEP FLA changes (CHANGE_0080) — proven beneficial
# (no action needed)

# 6. Commit & push
git add -A
git commit -m "Surgical revert: v19 minicpm.py + preprocess_model.py changes; baseline=v18 (a9f4d43cb)"
git push minicpm-src mixed_minicpm_cudagraph

# 7. Validate on fcloud (after user starts instance)
#    - rebuild sgl-kernel? NO — we didn't touch sgl-kernel
#    - sync to fcloud + restart server
#    - run accuracy + speed (Test 12 reference: S1=121.71, S8=44.09, Smax=35.86, acc≥80%)
```

---

## Awaiting your decisions before executing the revert

1. **eval_model_001.py policy**: strict (full revert) or pragmatic (keep additive opt-in flags)?
2. **prepare_env.sh**: leave HEAD as-is (Test 20 best combo) or restore exact v18 settings? Recommended: **leave as-is and revisit in M3** (long-context-faithful local set will give us proper data to choose).
3. **Confirm to execute** the revert commands above (only minicpm.py + `preprocess_model.py` if you pick "leave prepare_env" + "pragmatic eval"), commit, and push?

Once you decide, I'll also update the plan doc with these refined findings (FP4 KV pool exists upstream → M2.0 simpler, M2.1 mostly eliminated).  



