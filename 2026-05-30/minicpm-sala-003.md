The fix works. **65% MCQ accuracy** (13/20), avg_out=13,748 — the GLA state corruption is eliminated. Runaway generation is gone.

Let me pause the instance and create the required documentation: 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Starting: *Create CHANGE_0155 docs + update tracking* (5/6)

Pause failed (console JWT expired) — please pause the instance manually. Now creating the bilingual docs and updating tracking:  

Searched for files matching `**/docs/soar_2026_changes/CHANGE_01*.md`, 20 matches

Searched for text `Test 14|Test 13|Stage 3` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), 7 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#68-68), lines 68 to 90

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#74-74), lines 74 to 100

Now I'll add the Stage 3a GLA fix test row to the tracking file, then create the bilingual change docs: 

Replacing 1 lines with 2 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```
Now create the bilingual CHANGE_0155 Stage 3a doc:  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_stage3a_gla_fix.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_stage3a_gla_fix.zh.md)

Now commit the docs and push:  

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0155_stage3a_gla_fix.en.md docs/soar_2026_changes/CHANGE_0155_stage3a_gla_fix.zh.md docs/soar_2026_changes/TEST_RESULTS_TRACKING.md && git commit -m "docs(stage3a): CHANGE_0155 GLA fix + test tracking update

Add bilingual change docs for Stage 3a SimpleGLA initial_state fix.
Update TEST_RESULTS_TRACKING with Stage3a-GLA-fix row:
- MCQ 65.00% (13/20), avg_out=13,748, runaway generation eliminated.
- Commit 94f6ff6c6, ai-e7e98a7c52 fcloud, eager mode." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -4

Completed: *Create CHANGE_0155 docs + update tracking* (5/6)

**Stage 3a GLA fix is complete and verified.** Here's the summary:

---

**Root cause identified and fixed:**

`SimpleGLAAttnBackend.forward()` only loaded the recurrent state from cache when `is_decode()` or `_has_prefix_state()`. For `TARGET_VERIFY`, the mode is `is_extend()` and `extend_prefix_lens` is not set → `initial_state=None` (zero state) → every GLA layer ran from scratch, producing wrong logits AND writing a corrupted state back to cache. Every decode step after that cascaded into runaway `\n` generation.

**Fix** (one added condition in hybrid_linear_attn_backend.py):
```python
or forward_batch.forward_mode.is_target_verify()
```

**Results — Stage3a-GLA-fix (commit `94f6ff6c6`)**:
| Metric | Before | After |
|--------|--------|-------|
| MCQ accuracy | 0.00% (runaway) | **65.00%** (13/20) |
| avg_out | 55,648 tokens | 13,748 tokens |
| Runaway generation | YES | **NO** |

**⚠️ Action needed**: Please pause the fcloud instance manually — the console JWT is expired so `pause-instance` returned a 401 FAILED.

**Next: Stage 3b** — enable cuda-graph for TARGET_VERIFY and train/attach real K>1 Medusa heads for actual draft acceptance speedup. 



