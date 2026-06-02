Compacted conversation

Ran terminal command:  cd /home/oldzhu/sglang && git checkout 3a15a6de3 -- python/sglang/srt/speculative/medusa_worker.py benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py && git diff --stat HEAD

Ran terminal command:  cd /home/oldzhu/sglang && git add python/sglang/srt/speculative/medusa_worker.py benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py && git status --short

Ran terminal command:  cd /home/oldzhu/sglang && git commit -m "revert: CHANGE_0164 ndt=2 medusa_worker.py back to Stage 3a (3a15a6de3)" -m "Restore medusa_worker.py (both copies) to the Stage 3a baseline after the ndt=2 refactor produced ori_accuracy=15.13% / C=0 on fcloud (commit 4b442f421). Stage 3a (commit 3a15a6de3, Stage3a-force-dense row) = 78.40% acc / S1=204.86s is the best-known-good Medusa state; ndt=2 retry is paused pending verify-pipeline instrumentation (positions / seq_lens / tree_mask convention vs NGRAM)." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -10

Revert pushed as `a489d78d4`. Repo is back to Stage 3a (ndt=1) baseline.

**Status:**
- `e3a194641` — CHANGE_0164 catastrophic result docs (TEST_RESULTS_TRACKING, CHANGE_0164 EN/ZH, chat logs)
- `a489d78d4` — revert of `medusa_worker.py` (both copies) to commit `3a15a6de3` (Stage3a-force-dense)
- Untracked locally (not part of revert): train_medusa_head.py (unrelated pre-existing edit), full_server_logs.txt

**Not re-validated on fcloud** — Stage 3a was already verified at 78.40% acc / S1=204.86s in the `Stage3a-force-dense` row, so re-running the same code on the same model would be a paid no-op. fcloud remains paused.

**Suggested next steps (offline, no fcloud cost):**
1. Read `python/sglang/srt/speculative/ngram_info.py::NgramVerifyInput.prepare_for_verify` to learn whether it mutates `batch.seq_lens` and where it expects positions for bonus + drafts (off-by-one suspect #1).
2. Read `reconstruct_indices_from_tree_mask` to learn the canonical tree-mask layout NGRAM uses for ndt=2 — compare against the manual mask we built.
3. Once we have a concrete hypothesis for the bonus-position convention, write a single-step instrumentation script (eager mode, batch=1) that dumps `positions`, `seq_lens`, `req_to_token[req_pool_idx, seq_len-1:seq_len+ndt]`, and `tree_mask` for one NGRAM verify call vs one Medusa verify call. The delta will pinpoint the bug.

Want me to proceed with step 1 (read the NGRAM verify code) now?  


