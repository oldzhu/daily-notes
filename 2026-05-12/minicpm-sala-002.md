
ok so far the baseline is still dense + GPTQ + FP8 KV (Test 12), I am thinking 3 combinations:
1. using --attention-backend flashinfer + gptq + fp8 kv
2. using --attention-backend minicpm-flashinfer + --force-dense-minicpm + gptq+fp8 kv (test 12)
3. using --attention-backend minicpm-flashinfer + set long dense_len(need to modify sglang to support dense_len args) so that for short sequence using dense and for long sequence using sparse attention.
we haven't tested 1 and 2, for 1, would performacne a little bit better as now there is no minicpm as router to route the processing to flashinfer? for 2, not sure if it can get the same performance as dense for short sequences and get better performance than dense for long sequences.
=========
Quick analysis of the three options before any code change (per the proposal-first rule):

## Option 1 — `--attention-backend flashinfer` + GPTQ + FP8 KV

**Likely won't work / won't help.** Two problems:

- **Architecture mismatch.** MiniCPM-SALA has 24 lightning-attn (linear/Mamba-style) layers + 8 sparse_attention layers. The "router" inside `--attention-backend minicpm_flashinfer` isn't just a sparse-vs-dense dispatcher — it's a *hybrid linear-attn* backend that handles the lightning-attn layers via the FLA path. Pure `flashinfer` doesn't know about lightning-attn. Either model load fails or those 24 layers run wrong.
- **No real overhead to eliminate.** With `--force-dense-minicpm`, the per-request branch becomes a single boolean check at the head of each sparse layer's forward. It's nanoseconds per layer; you can't see it on a profile. There's no "router cost" to recover.

Recommendation: **skip option 1.** Expected gain ≈ 0%; risk of breakage ≈ high.

## Option 2 — `--attention-backend minicpm_flashinfer --force-dense-minicpm` + GPTQ + FP8 KV

**This IS Test 12** (the current baseline: S₁=121.71s, S₈=44.09s, S∞=35.86s, ori_acc=79.29%, C=1.0). Already tested. So we have nothing new to learn from running it again.

## Option 3 — `minicpm_flashinfer` + tunable `dense_len` + GPTQ + FP8 KV (NEW idea)

This is the genuinely interesting one. Two hypotheses to be honest about:

**Speed hypothesis (positive).** From CHANGE_0135_001 profile under BF16+sparse, sparse-FA itself is well-behaved (3.6–4.3% of GPU time, linear scaling). The reason BF16+sparse loses to Test 12 is the **GEMM precision**, not the attention algorithm. If we keep GEMMs on Marlin/INT4 (i.e. GPTQ for all layers) AND let only *very long* requests take the sparse path (`dense_len` set high, e.g. 32 768 or 65 536), sparse top-k *should* save attention I/O on those few long requests vs dense full-attention. So the theoretical curve is:
- short prompts → dense path → equal to Test 12
- long prompts → sparse path → potentially faster than Test 12

**Accuracy hypothesis (the risk).** Historically GPTQ + sparse routing collapsed to ~50% on the public set (the `sparse_qkv_w8` regression noted in `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`). Whether the collapse happens at **any** sparse-routed sample, or only at certain context lengths or when many layers route sparse simultaneously, has not been characterized. With `dense_len` cranked high, fewer samples route sparse → the failure mode might be small enough to stay above the 97% normalized-accuracy cliff, OR it might still collapse the few long samples to zero and cause the long-tail slowdown anyway.

**Proposed implementation (small, reversible):**

1. Add an optional CLI flag `--minicpm-sparse-dense-len <int>` to `prepare_env.sh` `SGLANG_SERVER_ARGS` and to the sglang server-args parser; have the model read it via env or args and override `hf_config.sparse_dense_len` at load time. (One file in srt, plus `prepare_env.sh`.)
2. Rule-compliance: this is a server-side runtime knob, no model-file edits, no eval-script edits. Compatible with submission.
3. Validation matrix (cheapest first; abort if step a fails):
   - **a. Sanity:** `dense_len = 524288` (effectively never sparse) → must reproduce Test 12 numbers exactly. Confirms the new flag doesn't perturb the dense path.
   - **b. Conservative threshold:** `dense_len = 65536` → only the longest tail goes sparse. Run accuracy + S₁/S₈/S∞.
   - **c. Aggressive threshold:** `dense_len = 16384` → most long context goes sparse. Run accuracy + speed.
   - Decision: pick the lowest `dense_len` that keeps normalized accuracy >99% (C=1.0) and watch whether speed improves on the higher tiers.

**Why this is worth doing:** it's the only path on the table that *can* beat Test 12 without re-quantizing or retraining. The downside risk is bounded — if step (b) accuracy fails we revert to Test 12 immediately.

**Want me to write this up as `CHANGE_0136_minicpm_sparse_dense_len_flag` (EN+ZH proposal) for approval?** That would include the exact files to touch, the env/CLI plumbing, and the validation commands. No code changes until you approve. 


