Searched for text `reconstruct_indices_from_tree_mask` (`**/sgl-kernel/**`), 12 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md#1-1), lines 1 to 40

Read [](file:///home/oldzhu/sglang/sgl-kernel/python/sgl_kernel/speculative.py#80-80), lines 80 to 110

Read [](file:///home/oldzhu/sglang/sgl-kernel/tests/speculative/test_ngram_utils.py#1-1), lines 1 to 75

Found the test which makes the kernel convention concrete: positions[0] = seq_lens (not seq_lens-1). Now I can write the analysis. 

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md#40-40), lines 40 to 200

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md#200-200), lines 200 to 238

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md#170-170), lines 170 to 195

Now appending the post-revert root-cause analysis to CHANGE_0164 docs, and creating the follow-up proposal. 

Replacing 14 lines with 146 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md)


```

```

```

```
Now creating the corrected ndt=2 proposal doc pair. 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_medusa_k1_positional_offbyone_fix.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_medusa_k1_positional_offbyone_fix.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md docs/soar_2026_changes/PROPOSAL_medusa_k1_positional_offbyone_fix.en.md docs/soar_2026_changes/PROPOSAL_medusa_k1_positional_offbyone_fix.zh.md && git commit -m "docs: CHANGE_0164 root-cause analysis + PROPOSAL positional off-by-one fix" -m "Post-revert offline analysis of the ndt=2 catastrophe. Reading ngram_info.py, ngram_worker.py, and test_ngram_utils.py established: (1) NgramVerifyInput is generic tree-verify infrastructure (Medusa reuse is correct); (2) prepare_for_verify does NOT mutate seq_lens; (3) kernel convention is positions[0]=seq_lens (root at next free slot). The actual root cause is a pre-existing positional invariant violation: we feed output_ids[-1] (token at conceptual position seq_lens-1) at position seq_lens, making KV slot k store the wrong token id with positional embedding k. ndt=1 tolerates this (~0.6 pt regression); ndt=2 amplifies through target_predict[1] and the bonus chain into catastrophic drift. Documented in CHANGE_0164 + new PROPOSAL_medusa_k1_positional_offbyone_fix sketching the seq_lens-decrement fix and canonical reconstruct_indices_from_tree_mask use." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Pushed as `7e06959ad`. Summary of what's now in the docs for your review:

## Key analytical conclusions

1. **`NgramVerifyInput` is misnamed** — it's generic tree-verify infrastructure; Medusa reusing it (as Stage 3a does) is intentional and correct.
2. **`prepare_for_verify` does NOT mutate `seq_lens`** — that pre-revert hypothesis is dead.
3. **Kernel convention pinned by test_ngram_utils.py:** for `seq_lens=12, ndt=4`, output is `positions=[12,13,13,14]`. The root sits at `seq_lens` (the **next** free slot), not `seq_lens-1`. So our hand-rolled `positions = seq_lens + arange(ndt)` was numerically correct.
4. **Real root cause** = pre-existing positional invariant violation, present even in Stage 3a:
   - sglang invariant: KV slot `k` ↔ token at conceptual position `k` in `origin_input_ids ++ output_ids`.
   - `output_ids[-1]` is at conceptual position `seq_lens - 1`, but we feed it at position `seq_lens`. So slot `seq_lens` gets KV with `positional_embedding = seq_lens` but `token_id = wrong-position token`.
   - NGRAM doesn't have this bug because its `input_ids[0]` is a *fresh n-gram prediction* for slot `seq_lens`, not a re-feed of the previous bonus.
   - ndt=1: model is robust → only ~0.6pt regression (v23 78.71 vs v22 79.29).
   - ndt=2: amplifies multiplicatively via `target_predict[1]` + cascading bonus chain → 15.13% catastrophe.
5. **`accept_rate=0.73` was misleading** — it measured tree-walk structural correctness against a *corrupted* `target_predict[0]`, not against the true argmax.

## Proposed fix (two-step, ndt=1 first)

- **Step 1** (independent value): decrement `seq_lens` by 1 before `prepare_for_verify` for ndt=1. Restores byte-equivalence to dense decode; recovers the 0.6pt v23 regression on its own.
- **Step 2**: same decrement + bump `ndt=2`, but use `reconstruct_indices_from_tree_mask` (kernel call, same as NGRAM) instead of hand-rolling `positions/retrive_*`. Trained head provides slot `seq_lens` draft.
- **Main risk**: page-pool semantics when we overwrite an already-allocated slot. Must read `get_src_tgt_cache_loc` + `_free_cache` before coding.
- **Pre-flight gates**: standalone single-step Python script that dumps `(positions, seq_lens_in/out, req_to_token slice, predicts, accept_length)` and compares against an NgramWorker step with a forced identical draft. No fcloud until this passes locally.

Docs committed (`7e06959ad`): CHANGE_0164 EN, CHANGE_0164 ZH, PROPOSAL EN, PROPOSAL ZH. Awaiting your review.  



