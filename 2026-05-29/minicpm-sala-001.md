Good strategic question — you're right that this matters before Stage 3. My recommendation: **fix cuda-graph for Stage 2 first, then go to Stage 3.** Three reasons:

## 1. Stage 3 needs cuda-graph anyway

Stage 3 verify path runs `TARGET_VERIFY` decode with `draft_token_num=2` per seq. If cuda-graph is broken for the spec path, Stage 3 won't beat Test 12 either — Medusa's typical 1.5–2.0× acceptance speedup gets eaten by the same eager-mode tax we see now. The cuda-graph fix is on the critical path either way; doing it now means Stage 3 lands on a fair baseline.

## 2. Without it we can't tell if Stage 3 actually works

Today's signal is muddied: "Stage 3 S1 = X" doesn't tell us if Medusa is winning or losing because the −65% eager-mode floor dominates. With cuda-graph fixed in Stage 2, we get a clean number: "pure pass-through ≈ Test 12 ± a few %", and then any Stage 3 delta is attributable to Medusa itself.

## 3. The fix is likely small

Root cause hypothesis (from the symptom `shape [1] doesn't match broadcast shape [7]`): cuda-graph capture/dispatch uses the **scheduler-level** `spec_algorithm == MEDUSA` to pad per-seq token count to `num_draft_tokens=2`, but our pass-through only feeds 1 token/seq to the target. Two candidate fixes (cheapest first):

- **A. Flip `batch.spec_algorithm = NONE` earlier** — currently we flip it inside `MedusaWorker.forward_batch_generation`, but cuda-graph runner reads `batch.spec_algorithm` *before* the worker is called (during `get_model_worker_batch` / graph key lookup in `cuda_graph_runner.py`). Moving the flip into `get_model_worker_batch` or doing it in `event_loop_normal` right after `run_batch` resolves cuda-graph dispatch.
- **B. Special-case Medusa-pass-through in cuda-graph runner** — branch on `spec_algorithm == MEDUSA` + `draft_token_num=1` and use the non-spec graph key. Slightly more invasive.

Both are 1–2 file changes, no new kernels. Worst case, if neither works in a day, we revert and proceed to Stage 3 with eager — but I'd budget that effort first.

## Plan if you agree

1. **CHANGE_0155 §14**: investigate cuda-graph dispatch for Medusa pass-through. Try fix A first (move the flip earlier). 
2. Re-run accuracy + S1 with cuda-graph + torch.compile re-enabled.
3. Target: S1 within +5% of Test 12 (i.e. ≤ 128s). Accept rate is still 0 (pass-through), so any remaining gap is pure framework overhead.
4. **Then** start Stage 3 proposal on a clean baseline.

If S1 fix succeeds → Stage 3 has 121s budget to play with.
If S1 fix fails → we still learned that the cuda-graph path needs a Stage 3-level redesign (graph capture under `TARGET_VERIFY`), which is useful Stage 3 scoping info.

**Do you want me to proceed with fix A (move `spec_algorithm=NONE` flip earlier)?** I'd start the fcloud instance, patch + push, then run S1-only first (5 min) before paying for full accuracy.
