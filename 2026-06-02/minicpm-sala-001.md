Good question. Let me check whether the v23 submission actually invokes `_forward_verify_k1` or stays in pass-through mode:     

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#1-1), lines 1 to 270

Found it. Let me verify my understanding by re-reading `prepare_for_verify` placement logic and the actual seq_lens accounting:  

Searched for regex `draft_token|seq_lens|prepare_for_verify|NgramVerifyInput|bonus` (`**/python/sglang/srt/speculative/ngram_worker.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#121-121), lines 121 to 255

Yes — this is the same fix. Let me walk through the deep analysis:

## Deep-dive: does v23 actually corrupt outputs?

### What the v23 submission does at runtime

Looking at `python/sglang/srt/speculative/medusa_worker.py:228-243`:

```python
def forward_batch_generation(self, batch):
    if batch.forward_mode.is_extend():
        # EXTEND: passthrough, no verify
        batch.spec_algorithm = SpeculativeAlgorithm.NONE
        ...
    # DECODE: runs _forward_verify_k1 (verify path)
    return self._forward_verify_k1(batch)
```

The "pass-through" comments in `prepare_env.sh` describe the original CHANGE_0155 Stage 2 worker. The current Stage 3b code **does** invoke the verify path on every decode step. The "Stage 3a fallback" sets `draft = req.output_ids[-1]` when no head is loaded.

### The subtle corruption

Tracing one decode step (Stage 3a, no head, K=1):

Before forward:
- `req.output_ids` ends with token `T_N` (appended last step)
- `seq_lens[i] = N+1` (already incremented to include `T_N`; KV cache has `H(T_N)` at position `N`)
- `draft = T_N` (= `output_ids[-1]`)

`prepare_for_verify`:
- `end_offset = seq_lens + 1 = N+2`
- `assign_req_to_token_pool(seq_lens=N+1, end_offset=N+2, out_cache_loc=…)` allocates a **new** KV slot at position `N+1` for the draft
- `batch.input_ids = [T_N]` (the duplicate)

Forward pass:
- KV cache now has: `…, H(T_N) at pos N, H(T_N_duplicate) at pos N+1`
- Attention at position `N+1` attends to `[prefix … , T_N (pos N), T_N (pos N+1)]`
- Produces `logits[pos N+1]` = next-token distribution **conditioned on `prefix + T_N + T_N` (duplicated)** instead of the correct `prefix + T_N`

Verify kernel (since `draft_token_num=1`, no children to walk):
- `accept_token_num = 0`, root always accepted
- `predicts[root] = argmax(logits[pos N+1])` — this is the **corrupted** next-token prediction
- `req.output_ids.append(predicts[root])` ← corrupted token enters the output sequence
- `seq_lens.add_(0 + 1) = N+2`

So every Stage 3a decode step appends a token sampled from a slightly wrong distribution. The wrong-context bias is small (one duplicate of the last token in a possibly-very-long context), but it's there and is consistent with the empirical observation:
- Test 12 baseline (no Medusa): 79.29%
- Stage 3a (Medusa on, force-dense): 77.87% — **−1.4pt**
- v23 official submission: 78.71% — **−0.6pt**

The −0.6 to −1.4pt regression is exactly what you'd expect from a small repetition-toward-last-token bias acting on every decode step. MCQ tasks (deterministic single-letter answers) are most affected; cwe (repetition tasks) is largely insensitive or slightly helped by the bias.

### Why this matters

1. **v23 submission IS silently corrupting outputs.** The corruption is small enough to look like noise but is structural and consistent.
2. The "pass-through" claim in `prepare_env.sh` Stage 2 comments is stale — Stage 3a/3b is not pass-through; it's running a broken verify path.

### Does this impact the refactor?

**No — it's the same fix.** Both the speed bug and the correctness bug have the same root cause: `draft_token_num=1` with no separate bonus position. The proposed refactor to `draft_token_num=2` with `[bonus=last_output, draft=speculation]` simultaneously:

| Concern | Current (broken) | After refactor |
|---|---|---|
| Bonus position | absent — `draft=last_output` duplicated into new slot | explicit at position 0, `seq_lens` decremented by 1 before prepare so bonus is the "new" position (NOT a duplicate) |
| Speed | verify kernel walks 0 children → no acceptance possible | walks 1 child → trained-head draft can be accepted |
| Correctness | next-token argmax conditioned on `prefix + T_N + T_N` | conditioned on `prefix + T_N` (correct) |

This means:
- **One refactor, two bugs fixed.** No need for sequential cycles.
- **No concurrent work needed** — there's nothing additional to investigate; the analysis is complete.
- The next fcloud cycle gets us:
  - Accuracy: should improve from 77.87% → ~79.3% (back to baseline) regardless of head presence
  - Speed: improves only if trained head accepts (~99% offline match suggests ~1.5-1.7x on decode-heavy workloads)

### Additional implementation detail discovered

Looking at `ngram_worker.py:39`: stock spec workers use `draft_token_num = server_args.speculative_num_draft_tokens` directly. `prepare_env.sh` already sets `NUM_DRAFT_TOKENS=$(( SOAR_SPEC_MEDUSA_HEADS + 1 ))` = 2 for K=1. **Our MedusaWorker incorrectly overrides this to `num_heads` (=1)**, dropping the bonus position. The fix is partially: change `self.draft_token_num = self.num_heads` → `self.draft_token_num = server_args.speculative_num_draft_tokens` (= num_heads + 1), and add the seq_lens decrement + bonus token to the verify input.

### Recommendation

Proceed with the single refactor I proposed. The proposal already covered §3a "[last_output, draft]" layout — it just needs an explicit "decrement `batch.seq_lens` by 1 before `prepare_for_verify`, then `accept_length+1` re-increments correctly after verify" step that I'll add to the proposal before implementing.

Awaiting your approval to:
1. Update `PROPOSAL_medusa_stage3b_k1_draft_token_num_2` with the seq_lens-management detail (no fcloud needed)
2. Implement the refactor in medusa_worker.py (both copies)
3. Resume fcloud, restart with v2 head, measure accept_len and accuracy (one cycle, ~30-45 min)

Expected outcomes from one cycle:
- accuracy returns to ≥ 79% (vs current 77.87% Stage 3a)
- if head is well-aligned, accept_len → ~1.9 and S1 drops from 216s → ~120-140s
- if head doesn't generalize well from offline 99.91% match to online, accept_len ~1.0-1.2 and S1 mostly unchanged — but accuracy fix is still validated

Permission to proceed?     Compacted conversation


