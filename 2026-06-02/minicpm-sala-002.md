Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#100-100), lines 100 to 280

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#280-280), lines 280 to 466

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_info.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_info.py#200-200), lines 200 to 320

Searched for regex `seq_lens.add_|seq_lens_cpu|verify\(` (`**/python/sglang/srt/speculative/ngram_info.py`), 18 results

Searched for text `VerifyTreeGreedy` (`**/sgl-kernel/csrc/speculative/eagle_utils.cu`), 2 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/speculative/eagle_utils.cu#272-272), lines 272 to 370

Searched for regex `prepare_for_decode|positions = batch.seq_lens|input_ids = batch.output` (`**/python/sglang/srt/speculative`), no results

Searched for text `def prepare_for_decode` (`**/python/sglang/srt/managers/schedule_batch.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/schedule_batch.py#1939-1939), lines 1939 to 2010

Searched for regex `draft_token\[|topk_index|verified_id|bonus|retrive_index|seq_lens - 1` (`**/python/sglang/srt/speculative/eagle_info.py`), 40 results

Searched for regex `num_draft_tokens|verified_id|build_tree|bonus` (`**/python/sglang/srt/speculative/eagle_worker.py`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_worker.py#555-555), lines 555 to 605

Searched for regex `seq_lens\.sub|seq_lens -|seq_lens_cpu -|prepare_for_verify|prepare_for_decode` (`**/python/sglang/srt/speculative/eagle_info.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_info.py#100-100), lines 100 to 175

Execution complete: Diff medusa_worker.py copies

Ran terminal command:  diff -u /home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py /home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py

Now I have a complete picture. Let me also verify the proposal exists and what it says, then implement: 

Searched for regex `seq_lens|tree_mask|retrive_index` (`**/docs/soar_2026_changes/PROPOSAL_medusa_stage3b_k1_draft_token_num_2.en.md`), 11 results

Created 5 todos

Now implement the refactor. First the upstream copy: 

Replacing 14 lines with 24 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)


```

```
Now rewrite `_forward_verify_k1`:  

Replacing 182 lines with 201 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)


```

```


Checked [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py), no problems found

Now sync to the demo_sala copy:  

Execution complete: Sync medusa_worker to demo_sala

Ran terminal command:  cp /home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py /home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py && diff -q /home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py /home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py

Starting: *Write CHANGE_0164 continuation doc (EN+ZH)* (3/4)

Searched for files matching `**/docs/soar_2026_changes/CHANGE_0164*`, no matches

Searched for files matching `**/docs/soar_2026_changes/CHANGE_0164*.md`, no matches

Execution complete: List Medusa docs

Ran terminal command:  ls /home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_016* /home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_medusa* /home/oldzhu/sglang/docs/soar_2026_changes/*medusa* 2>/dev/null

Execution complete: List exact medusa filenames

Ran terminal command:  ls -1 /home/oldzhu/sglang/docs/soar_2026_changes/ | grep -Ei 'medusa|CHANGE_016'

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md)

Starting: *Commit and push to minicpm-src* (4/4)

Execution complete: Commit and push to minicpm-src

Ran terminal command:  cd /home/oldzhu/sglang && git status --short

Ran terminal command:  git add python/sglang/srt/speculative/medusa_worker.py benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md

Ran terminal command:  git status --short

Ran terminal command:  git commit -m "CHANGE_0164: Medusa Stage 3b K=1 refactor to ndt=2 canonical layout" -m "Fixes both the speed bug (kernel has no children to validate with ndt=1) and the correctness bug (predicts[root] writes target_predict at duplicated-token position, biasing every Medusa decode step). Canonical EAGLE/NGRAM layout: position 0 = bonus = last committed token, position 1 = speculative draft. Use CaptureHiddenMode.FULL and take position-0 hidden for head forward and offline dump. See docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.{en,zh}.md."

Ran terminal command:  git log --oneline -3

Ran terminal command:  git push minicpm-src mixed_minicpm_cudagraph

Completed: *Commit and push to minicpm-src* (4/4)

Pushed `4b442f421` to `minicpm-src`. Refactor summary:

**Code changes** (both medusa_worker.py and the synced medusa_worker.py):
- `self.draft_token_num = self.num_heads + 1` (=2 for K=1) — matches the canonical EAGLE/NGRAM bonus+draft layout that `prepare_env.sh` already advertises via `NUM_DRAFT_TOKENS=$(( SOAR_SPEC_MEDUSA_HEADS + 1 ))`.
- `_forward_verify_k1` rewritten: position 0 = `output_ids[-1]` (bonus), position 1 = head prediction (or duplicate fallback). Linear-chain `retrive_*` arrays, causal `tree_mask` per NGRAM `USE_FULL_MASK` convention.
- Switched from `CaptureHiddenMode.LAST` to `FULL`; bonus position-0 hidden feeds both the head and the offline dump (same context as v1 training).
- Dropped the dead `CHANGE_0160` bonus-zeroing loop (it only fired with `accept_length≥1`, never reachable under ndt=1; ndt=2 path uses standard `_free_cache`).

**Why this fixes both bugs**:
- Speed: kernel now has a child to walk → trained head's correct draft can be accepted → up to 2 tokens/step.
- Correctness: regardless of acceptance, `predicts[0] = target_predict[0]` is the model's argmax conditioned on `prefix + T_N` (correct), no longer on the duplicated-token context. Stage 3a fallback now also produces standard-decode-equivalent output.

**Docs**: CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md, CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md.

**Next step (awaiting your approval)**: resume fcloud, sync, restart server, run accuracy + S1/S8/Smax. Two runs:
1. `SOAR_MEDUSA_HEAD_PATH=""` → expect acc back to ~79 % (correctness check)
2. With v2 head → measure `accept_len` and S1 (speed check)

Ready when you say go. 



