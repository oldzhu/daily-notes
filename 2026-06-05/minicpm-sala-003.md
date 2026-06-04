Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

fcloud paused. Now write §5.2 docs. Let me read current CHANGE_0165 to understand structure. 

Ran terminal command:  wc -l /home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md /home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md && tail -40 /home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md

Searched for regex `5\.1|## 5` (`**/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md`), 4 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md#96-96), lines 96 to 175

Now I'll append §5.2 to both EN and ZH docs.  

Replacing 3 lines with 63 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md)


```

```
Now the ZH version. Let me check its tail structure first.  

Searched for regex `## 6|Commits this iter|本轮提交|disable graph` (`**/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md`), 2 results

Replacing 3 lines with 63 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md#140-140), lines 140 to 155

Replacing 3 lines with 63 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md)


```

```
Commit and push docs.   

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md && git commit -m "preflight: CHANGE_0165 §5.2 iter2 result (13→5 diffs, isolated to seq_lens off-by-one)" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Iter 2 complete and documented. Summary for user:

## Iter 2 result — pre-flight worked as designed

**Diff count: 13 → 5** (commit 89a30d5d0 for code, c29de9b92 for §5.2 docs)

Layout fix succeeded — these 8 fields are now EQUAL between NgramWorker and MedusaWorker:
- `draft_token_num` (1→2)
- `input_ids`, `spec_draft_token` (now `[11225, 0]` on both)
- `spec_custom_mask` (18 bool, USE_FULL_MASK layout)
- `spec_retrive_index` / `next_token` / `next_sibling` (kernel-built, identical tree)
- `logits_shape` (2, 73448)

`post_verify` remains **4×EQUAL** (accept_length, accepted_indices, next_token_ids, num_accepted_tokens).

**Single remaining root cause** — `seq_lens` is +1 on MedusaWorker (8 vs 7). This cascades into 4 derived diffs (`seq_lens_cpu`, `spec_positions`, `out_cache_loc`, `logits_argmax[1]=59320 vs 72`). The pre-flight harness has now isolated the long-standing CHANGE_0160 / CHANGE_0161 positional bug to **one field in one phase** — no more layout-cascade ambiguity.

**Three hypotheses for the +1** documented in §5.2:
1. **H1 (most likely)** — Stage 3a bonus-position KV write commits before `prepare_for_verify` runs
2. **H2** — `prepare_for_verify` increments `seq_lens` prematurely
3. **H3** — Initial decode-extend commits two slots

**Iter 3 plan (needs your "go")**: Add a 4th dump phase `pre_prepare_for_verify` to disambiguate H1/H3 vs H2, then surgically fix the offending +1.

fcloud paused. Awaiting your "go" for iter 3. 



