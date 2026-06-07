Excellent question — this is exactly the right thing to challenge. Here's the actual evidence we have, and where it's thin.

## Why NGRAM was chosen as the baseline

NGRAM is the *closest analog* to MEDUSA K=1 in sglang:
- Both go through the same `TARGET_VERIFY` pipeline (`prepare_for_verify` → `forward_extend` → `verify`).
- Both use the same `NgramVerifyInput` / `MedusaInput` spec_info objects with similar tensor layouts.
- Both run on the same attention backend, KV pool, scheduler.
- The crucial difference: NGRAM picks the draft from a string-match table; MEDUSA picks it from a learned head. In our preflight, MEDUSA was deliberately neutered to `head_pred=0` so it always drafts `output_ids[-1]` — the **same draft NGRAM produces** in a K=1 zero-state scenario.

So the *theory* was: same draft + same pipeline → same output. Any byte-diff in the dumps localizes a MedusaWorker-vs-NgramWorker code-path difference.

## Direct evidence NGRAM works on this stack

This is where we should be honest — **the evidence is weaker than ideal**.

| Evidence | Strength |
|----------|----------|
| **NGRAM-reprobe-quick** (2026-05-12, commit `a91a0a068`): ran NGRAM verify end-to-end on 20 mcq samples after the GLA `is_target_verify` patch. Result: **0.00% MCQ, avg_out_tokens=44,916** — exactly the same runaway pattern we just saw with MEDUSA. | **Negative** — NGRAM also fails on this config. |
| **Stage3a-GLA-fix** (2026-05-12, commit `94f6ff6c6`): ran MEDUSA (K=1, eager) on 20 mcq, got **65% MCQ, avg_out=13,748** — the GLA fix did rescue eager-mode MEDUSA. | NGRAM was not re-run after the fix at scale, only the kernel patch was reused. |
| NGRAM in sglang upstream is a maintained feature, used by other models successfully. | Indirect — those models don't use SALA's hybrid GLA+std-attn+FP8 KV+force-dense combination. |
| Our preflight uses the **same engine binary** for NGRAM and MEDUSA. Bugs in the shared scheduler/KV path would affect both equally. | This is what makes the diff *useful* — but only as a *relative* check. |

## What this actually means

**NGRAM is not a verified-correct baseline on our stack.** It's a *behavioral twin* of MEDUSA: most likely both are wrong in the same way, both produce the same garbage. The preflight diff doesn't prove either path is correct — it only proves they're **equally (in)correct on iter 1**.

Look back at the NGRAM-reprobe-quick row in TEST_RESULTS_TRACKING.md:
> avg_out=44,916 tokens/sample (vs ~10–50 expected). Model generates ~45 K tokens per MCQ question hitting the 65,536 max_tokens limit.

That is *the same failure mode we just observed with MEDUSA iter 4* (avg_out=55,816, mcq=0%). The diagnosis at the time blamed GLA state mismatch; the CHANGE_0157 patch fixed eager-mode MEDUSA decode but the **NGRAM end-to-end was never re-validated after that patch**.

## So what is the true baseline?

The only baseline we have *positive end-to-end evidence* for is:

- **Stage 2-cgraph** (commit `46553947b`): MEDUSA worker instantiated but pass-through *without* TARGET_VERIFY. ori_accuracy=80.11%, C=1.0. This proves the MedusaWorker plumbing is fine — the bug is specifically in the TARGET_VERIFY path.
- **Test 12 / v18-revert** (`8d1e4d12b`): NgramWorker disabled entirely, classic decode. 79.29%, C=1.0.

Neither one exercises `TARGET_VERIFY`. **TARGET_VERIFY itself is the broken thing** on our hybrid GLA + force-dense + FP8 KV + GPTQ + torch-compile stack — for both NGRAM *and* MEDUSA.

## Implications for the diff strategy

The preflight approach was built on a false premise: that NGRAM is known-good. It isn't. The 0-diff we achieved means "MEDUSA is now the same kind of broken NGRAM is" — which is not progress toward correctness.

To make the preflight strategy actually work, we need a different baseline. Three options:

1. **Validate NGRAM first.** Run a real 150-sample accuracy eval with NGRAM enabled on the same config we used for iter 4. If NGRAM ≈ 79% C=1.0, then NGRAM is a valid baseline and the MEDUSA preflight diff was indeed meaningful (and the remaining bug is something the dumps missed — i.e. the multi-iter / in-kernel hypotheses still apply). If NGRAM is also catastrophic, the baseline strategy needs to be scrapped and replaced.

2. **Diff against the working pass-through path** (Stage 2-cgraph). This means diffing "MEDUSA with verify enabled" against "MEDUSA without verify" at corresponding points (after each decode step). The state SHOULD differ (because verify accepts multiple tokens), but the *committed-token sequence* given identical sampling should match for K=1 head_pred=0 (because the draft is always `output_ids[-1]`, which the model would have produced next anyway). This is a stronger correctness check than NGRAM-vs-MEDUSA.

3. **Diff against a CPU reference** for a tiny prompt: run the same prompt through HF transformers on CPU and compare the first N committed token ids. This is the gold standard — but expensive and only feasible for very short prompts.

Option (1) is the cheapest sanity check and should arguably be done before anything else. It costs ~50 min of fcloud time and definitively tells us whether the preflight strategy itself has any hope, or whether NGRAM is also dead-on-arrival on this stack.

## Bottom line

Your instinct is correct: the entire iter 4 effort assumed NGRAM was a trustworthy oracle. The one piece of evidence we have on this exact stack (NGRAM-reprobe-quick, 0% MCQ) actually contradicts that assumption. The strongest argument for now is: **TARGET_VERIFY is broken on our hybrid stack regardless of which worker drives it**, which is consistent with H3 in §5.7 (GLA state drift on the EXTEND path of verify, not yet patched).

If we proceed with spec decoding, validating NGRAM end-to-end is the prerequisite step. If NGRAM also fails at full eval, option (3) — ship Stage 2-cgraph as v23 and abandon Stage 3a — becomes nearly mandatory.
