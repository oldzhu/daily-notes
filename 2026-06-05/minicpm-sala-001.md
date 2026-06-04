I'll set up this properly per project rules: write the plan doc first (CHANGE_0165 EN+ZH), then get your explicit approval before any fcloud action. The diff-iteration protocol and per-diff decision template will live in the doc itself, so every subsequent iteration just appends to the same file pair.  

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes)

Searched for files matching `**/docs/soar_2026_changes/chat/CHAT_medusa*`, 20 matches

Searched for files matching `**/speculative/ngram_worker.py`, 2 matches

Searched for files matching `**/speculative/medusa_worker.py`, 2 matches

Searched for regex `def |prepare_for_verify|seq_lens|positions|draft_token_num|verify_tree_greedy|reconstruct_indices` (`**/python/sglang/srt/speculative/ngram_worker.py`), 20 results

Searched for regex `def |prepare_for_verify|seq_lens|positions|draft_token_num|_forward_verify|CaptureHiddenMode` (`**/python/sglang/srt/speculative/medusa_worker.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#60-60), lines 60 to 120

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#230-230), lines 230 to 340

Now I have enough to write the proposal. Let me also peek at NgramWorker's verify step briefly for the comparison spec. 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#120-120), lines 120 to 255

Enough context. Writing the proposal docs now. 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md)

Now the chat log per project rule, then commit.  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_medusa-preflight-plan_20260514_1100.en.md)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_medusa-preflight-plan_20260514_1100.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md docs/soar_2026_changes/chat/CHAT_medusa-preflight-plan_20260514_1100.en.md docs/soar_2026_changes/chat/CHAT_medusa-preflight-plan_20260514_1100.zh.md && git commit -m "medusa: CHANGE_0165 pre-flight diff plan (proposal)" -m "Plan a single-process pre-flight script that runs NgramWorker (reference) and MedusaWorker (test) end-to-end one verify step each, dumps spec_info/seq_lens/positions/tree_mask/retrive_*/out_cache_loc/logits-argmax/accept_length, and produces a structured field-by-field diff. Per-diff iteration protocol mandates a table with severity (critical/cosmetic/unknown) and decision (clean/keep/investigate) for every divergence. All iterations append to CHANGE_0165 as §5.N subsections. No code changes yet; awaiting user approval on script content + fcloud start-instance." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -20

Plan doc committed and pushed as `bd42903ec`.

## What I've put in place

CHANGE_0165 EN / CHANGE_0165 ZH holds the full plan. The two key bits you asked about:

**Pre-flight script outline** (§3): a single Python script run on fcloud that, in one process, runs prefill + one verify step with **NgramWorker** (`--speculative-algorithm NGRAM --speculative-num-draft-tokens 2`) as reference, then again with **MedusaWorker** (`--speculative-algorithm MEDUSA --speculative-num-medusa-heads 1 --speculative-num-draft-tokens 2`) as test, dumping at each step:
- `batch.seq_lens` (before and after `prepare_for_verify`)
- `spec_info.draft_token`, `positions`, `tree_mask`, `retrive_index`, `retrive_next_token`, `retrive_next_sibling`, `draft_token_num`
- `model_worker_batch.input_ids`
- `out_cache_loc` (KV slots written)
- post-forward `next_token_logits` argmax at each verify position
- `accept_length`, `next_token_ids`

Then phase C emits a structured field-by-field diff report. Cost per iteration: ~5–10 min of fcloud GPU time.

**Per-diff documentation template** (§4) — every diff iteration appends a §5.N subsection with a table:

| Field | A (Ngram) | B (Medusa) | Differs? | Severity | Decision | Why |
|---|---|---|---|---|---|---|

with explicit decision rules:
- **severity**: `critical` (changes forward semantics → must clean), `cosmetic` (no effect → may keep), `unknown` (probe further)
- **decision**: `clean` (modify Medusa to match Ngram), `keep` (justify intentional divergence), `investigate` (add diagnostic, re-run)

This guarantees that every diff is recorded **with its analysis and the decision rationale**, exactly as you asked — whether we clean it or keep it.

## Approval gates (per project rules)

Before any fcloud action I need explicit "go" on:

1. **The plan in CHANGE_0165** — does the field list / diff protocol / iteration cadence look right? Anything to add or remove before I write code?
2. **Permission to write the pre-flight script** (`benchmark/soar/demo_sala/preflight_medusa_vs_ngram.py`). I'll show you the script content for review before any fcloud run.
3. **Permission to `start-instance`** and run iteration 1 (after script approval).

Tell me which of these you're approving and I'll proceed. If you want changes to the plan first (e.g., add more fields, different iteration shape), say which and I'll revise the doc before writing the script.  



