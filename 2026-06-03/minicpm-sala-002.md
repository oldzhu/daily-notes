Searched for files matching `**/speculative/ngram*.py`, 4 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#80-80), lines 80 to 280

Searched for regex `class NgramVerifyInput|def prepare_for_verify|def verify` (`**/python/sglang/srt/speculative/ngram_info.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#280-280), lines 280 to 280

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_info.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_info.py#370-370), lines 370 to 450

Searched for regex `positions|tree_mask` (`**/python/sglang/srt/speculative/ngram_worker.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#120-120), lines 120 to 230

Yes — confirmed: **`NgramVerifyInput` is generic tree-verify infrastructure, not NGRAM-algorithm-specific.** Medusa using it is correct and intentional. Stage 3a already does this (and was the best-known-good Medusa state).

## Why it's the same thing

In sglang, speculative decoding is split into two stages:

1. **Draft generation** — algorithm-specific:
   - NGRAM: lookup last-n-token n-grams in the request's committed history.
   - Medusa: predict via trained head from last hidden state.
   - EAGLE: run a small draft model.
2. **Target verify** — shared infrastructure (this is what `NgramVerifyInput` provides):
   - Build draft tree (chain for K=1, real tree for K>1).
   - Run TARGET_VERIFY forward over the draft tokens with a `tree_mask` causal-attention pattern.
   - Walk the tree with `verify_tree_greedy` / `tree_speculative_sampling_target_only` to find the longest-accepted prefix.
   - Commit bonus token from target's argmax at the last accepted slot.
   - `batch.seq_lens.add_(accept_length + 1)` at the very end.

The class is poorly named (it predates Medusa support in this fork) — it's really `TreeVerifyInput`. `SpecInputType.NGRAM_VERIFY` is just a tag. `verify_tree_greedy` has nothing NGRAM-specific in it.

## What `prepare_for_verify` does (confirmed by reading ngram_info.py)

- Sets `batch.input_ids = self.draft_token` (length `bs * draft_token_num`).
- Allocates `out_cache_loc` for `bs * draft_token_num` new slots.
- Calls `assign_req_to_token_pool(..., batch.seq_lens, end_offset=seq_lens+ndt, ...)` which writes `req_to_token[req_pool_idx, seq_lens:seq_lens+ndt] = out_cache_loc`.
- **Does NOT mutate `batch.seq_lens`.** → My earlier "Hypothesis 3" (seq_lens mutation) was wrong.

`seq_lens` advances only inside `verify()` itself: `batch.seq_lens.add_(self.accept_length + 1)`, after the forward.

## Where the ndt=2 bug actually lives

Reading ngram_worker.py, NGRAM computes positions and tree_mask via the **`reconstruct_indices_from_tree_mask` sgl-kernel call** — not by hand:

```python
reconstruct_indices_from_tree_mask(
    tree_mask, batch.seq_lens,
    positions,            # mutable, written by the kernel
    retrive_index, retrive_next_token, retrive_next_sibling,
    bs, self.draft_token_num,
)
```

It takes a compact `(bs, ndt, ndt)` tree-shape mask, decodes per-node depth, and emits `positions[i, j] = seq_lens[i] + depth_of_node_j`. Only **after** that, when `USE_FULL_MASK=True`, NGRAM rebuilds the flat `(bs * ndt, seq_len + ndt)` causal mask for flashinfer.

Our ndt=2 refactor **bypassed `reconstruct_indices_from_tree_mask` entirely** and hand-rolled `positions = seq_lens + arange(ndt)` plus a manual `tree_mask`. For K=1 ndt=2 (a 2-node linear chain) the positions look numerically right, but:

- `retrive_index` / `retrive_next_token` / `retrive_next_sibling` semantics for the **bonus position** (root of the chain) are kernel-internal — they're not what we set (`retrive_index = arange(bs*ndt).view(bs,ndt)`, `retrive_next_token[:,0]=1`).
- The bonus token in NGRAM convention is **not** a separate draft position — it's recovered from `verify_tree_greedy`'s argmax at the accepted leaf, AFTER the verify walk. So putting "bonus = last committed token" at `positions[0]` and "draft = head prediction" at `positions[1]` corrupts the tree: the kernel thinks our chain is `root=bonus_token → child=draft_token`, but really `bonus_token` is a token the model has *already produced* on its previous step — re-feeding it as a draft makes the target re-predict from one-token-too-far back, then the verify kernel commits whatever argmax appears at slot 0 (which is the model's prediction of "the token after bonus_token" given a corrupted context). That's exactly the symptom we saw: `accept_rate=0.73` (kernel structurally happy) but garbage committed tokens (model never emits stop token → 64k runaway on every MCQ).

So the right fix for any future ndt=2 retry is to **call `reconstruct_indices_from_tree_mask`** (the same way NGRAM does) with a properly-shaped K=1 ndt=2 tree mask, instead of building positions/retrive_* by hand.

## Next plan (offline, no fcloud cost)

Stage 3a is back on `main`. I'll proceed offline as planned, no need to re-run fcloud:

1. ✅ Already confirmed: `NgramVerifyInput.prepare_for_verify` does not mutate `seq_lens` → drop that hypothesis from CHANGE_0164's "next investigation" list.
2. Document the **real** root cause (manual positions/retrive_* not equivalent to `reconstruct_indices_from_tree_mask`) in CHANGE_0164's revert section.
3. Sketch the corrected ndt=2 design: build a compact `(bs, ndt, ndt)` tree mask (lower-tri for K=1 chain), call `reconstruct_indices_from_tree_mask`, then expand to full mask. Save as a follow-up proposal — don't code until reviewed.

Want me to do (2) + (3) now (offline doc updates only)? 


