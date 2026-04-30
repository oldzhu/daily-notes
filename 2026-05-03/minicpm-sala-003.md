Ran terminal command:  cd /home/oldzhu/sglang && git status --short && echo "---" && git diff a9f4d43cb..HEAD --stat -- python/sglang/srt/models/minicpm.py benchmark/soar/demo_sala/preprocess_model.py

Ran terminal command:  cd /home/oldzhu/sglang && git checkout a9f4d43cb -- python/sglang/srt/models/minicpm.py benchmark/soar/demo_sala/preprocess_model.py && git status --short

Ran terminal command:  cd /home/oldzhu/sglang && git commit -m "Surgical revert to v18 baseline: minicpm.py + preprocess_model.py

Reverts CHANGE_0110 fold-scaling (minicpm.py, commits 96a4c6318/f373fbade/96304f9cd)
and CHANGE_0110 dense-attention GPTQ calibration (preprocess_model.py,
commits 30dc39371/08213639a) back to v18 commit a9f4d43cb.

Rationale: 3-way isolation tests (v19, v19-a, v19-c-FLA) showed the v19
minicpm.py changes are the prime suspect for the 3000s hang at sample 138 and
contribute to ~1.5pt accuracy regression vs v18. preprocess_model.py reverted
for cleanliness (does not affect already-quantized fcloud model).

KEPT: FLA chunk.py + chunk_delta_h.py (CHANGE_0080) — proven beneficial
(+1.75pt acc per v19-c-FLA test).
KEPT: prepare_env.sh (HEAD already at Test 20 best combo, post-v19).
KEPT: eval_model_001.py opt-in stratified-sampling flags (additive, default-None
preserves byte-identical behavior to upstream when unused; pragmatic carve-out
per user policy decision 2026-04-26).

Baseline: a9f4d43cb (v18, 2026-04-14)
Next iteration: M2 mixed FP8/NVFP4 KV cache (M2.0 ablation phase first)" && git log --oneline -3

Ran terminal command:  cd /home/oldzhu/sglang && git push minicpm-src mixed_minicpm_cudagraph

Surgical revert committed and pushed to `minicpm-src/mixed_minicpm_cudagraph`.

## Summary

| Step | Result |
|---|---|
| Revert minicpm.py to v18 (a9f4d43cb) | ✅ −62 lines |
| Revert preprocess_model.py to v18 | ✅ −160 lines |
| Commit | ✅ `8d1e4d12b` |
| Push to `minicpm-src` (NOT origin) | ✅ `89da4bf15..8d1e4d12b` |
| Keep FLA `chunk.py` + `chunk_delta_h.py` (CHANGE_0080) | ✅ unchanged (proven beneficial) |
| Keep `prepare_env.sh` (Test 20 best combo, post-v19) | ✅ unchanged |
| Keep `eval_model_001.py` opt-in flags (pragmatic) | ✅ unchanged |

**Branch state**: `mixed_minicpm_cudagraph` HEAD = `8d1e4d12b` = effectively v18 source baseline + (FLA improvements) + (Test 20 best server args) + (additive eval-script flags).

## Next steps (awaiting your green-light)

1. **Validate revert on fcloud**: when you start the instance, I'll run `fcloud_workflow.py sync && restart-server && wait-server && accuracy && speed --variant all` to confirm:
   - Accuracy ≥ v18 baseline (~80%)
   - No 3000s hang at sample 138
   - Speed matches/beats Test 12 reference (S1=121.71, S8=44.09, Smax=35.86)
2. **Update plan doc** with action 1+2 findings (FP4 KV pool already exists upstream → M2 simpler than originally scoped)
3. **Begin M2.0** (8-layer ablation harness) once revert is validated

Want me to:
- **(a)** wait for you to start fcloud, then run the revert validation tests?
- **(b)** in parallel right now (offline), update the plan doc with the FP4 upstream finding and start drafting `PROPOSAL_iteration_M2.0_kv_fp4_ablation.{en,zh}.md`?

Or just **(b)** for now and defer fcloud validation until you're ready?   


