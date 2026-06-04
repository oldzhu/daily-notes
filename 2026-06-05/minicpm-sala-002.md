Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#1-1), lines 1 to 400

Searched for regex `draft_token_num|reconstruct_indices|prepare_for_verify|tree_mask|positions|retrive_index|custom_mask|draft_token =|forward_batch_generation` (`**/python/sglang/srt/speculative/ngram_worker.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#1-1), lines 1 to 280

Now I have full picture. Let me check NgramVerifyInput for any extra detail to make the proposal precise. 

Searched for regex `def __init__|def prepare_for_verify|def verify|custom_mask|self\.draft_token` (`**/python/sglang/srt/speculative/ngram_info.py`), 20 results

Got enough. Now drafting the iter 2 proposal.  

Starting: *Iter2 proposal: detail cause + fix* (7/11)

# CHANGE_0165 — Iter 2 Proposal: detailed cause + fix

## 1. Cause (what §5.1 actually tells us)

**Root cause = one line of code.** Every one of the 13 differing fields in iter-1 cascades from a single decision in medusa_worker.py:

| Site | Stage 3a code | Effect |
|---|---|---|
| `MedusaWorker.__init__` | `self.num_heads = 1`, `self.draft_token_num = self.num_heads` → **`ndt = 1`** | Worker advertises 1 draft slot per request to the scheduler. |
| `MedusaWorker._forward_verify_k1`, `NgramVerifyInput(...)` | hard-coded `draft_token_num=1` (even when `--speculative-num-draft-tokens 2`) | The verify object also reports `ndt=1`, so `prepare_for_verify` allocates exactly `bs * 1` KV slots, expands `input_ids` to length `bs * 1`, etc. |

Why this matters semantically (not just shape):
- Speculative decoding's "chain of length `ndt`" really means **(base accepted token) + (draft predictions for the next `ndt - 1` positions)**, all of which the target model verifies in **one forward**.
- With `ndt = 1`, MedusaWorker only verifies the **base** token (`output_ids[-1]`). There is **no draft slot** for the trained Medusa head's prediction. So even with trained heads loaded (Stage 3b), the head output is computed → cached on `req._medusa_draft_token` → **never actually verified**. Verify keeps re-confirming the same already-accepted base token, accepting 0 bonuses (which matches the iter-1 dump: `accept_length=0`, `num_accepted_tokens=0`).
- With `ndt = 2` and the correct K=1 linear-chain metadata, the chain is `[output_ids[-1], head0_pred]`. The target forward returns logits for **2** positions; the verify walk checks whether the model's top-1 at position 0 equals position 1's draft. If so, **1 bonus token is accepted**, meaning **1 fewer forward per step on average** — that's the entire point of K=1 Medusa.

Cascade map for every other diff field, in plain English:

| Field | Reason for diff under iter-1 baseline | Will be clean after iter-2? |
|---|---|---|
| `draft_token_num` (2 vs 1) | direct: the constant in MedusaWorker | ✅ root fixed |
| `input_ids` shape (2 vs 1) | `prepare_for_verify` sets `batch.input_ids = spec_info.draft_token`, whose length is `bs * ndt` | ✅ shape becomes `(2,)` |
| `out_cache_loc` shape (2 vs 1) | `prepare_for_verify` reserves `bs * ndt` KV cache slots | ✅ shape becomes `(2,)` |
| `spec_draft_token` shape | same — `draft_token` is constructed at `bs * ndt` length | ✅ becomes `(2,)` `= [base, head_pred]` per req |
| `spec_positions` shape (2 vs 1) | result of `reconstruct_indices_from_tree_mask` over `bs, ndt`; with ndt=1 medusa builds a 1-element `seq_lens.clone()` instead | ✅ becomes `[7, 8]` per req — token-0 at current seq_len, token-1 one past it |
| `spec_retrive_index` (1,2 vs 1,1) | tree retrieval table shape is `(bs, ndt)` | ✅ becomes `(1, 2) = [0, 1]` |
| `spec_retrive_next_token` (1,2 vs 1,1) | for a linear chain of length ndt: indices `[1, 2, ..., -1]` | ✅ `[1, -1]` |
| `spec_retrive_next_sibling` (1,2 vs 1,1) | linear chain has no siblings | ✅ `[-1, -1]` |
| `spec_custom_mask` (18 vs 8) | full-mask layout per req is `(ndt, seq_len + ndt)` flattened. iter-1 medusa builds `ones(seq_len_i)` (single row) = 8 elements; ngram with `USE_FULL_MASK=True` does `(2, 7+2) = 18` | ✅ becomes 18 once we mirror Ngram's `USE_FULL_MASK` path |
| `logits_argmax` / `logits_shape` (2,V vs 1,V) | the forward produces one logits vector per draft position; with ndt=1 only one is produced | ✅ becomes `(2, V)` |
| `seq_lens` (7 vs 8) | side-effect: with `ndt=1` the engine has to issue **one extra plain decode** to consume the token that `ndt=2` would have consumed in the same verify step — bookkeeping offset, not an algorithmic bug | ✅ becomes 7 = 7 once both paths consume the same number of tokens per verify |
| `accept_length`, `accepted_indices`, `next_token_ids`, `num_accepted_tokens` | already equal in iter-1 — confirms the verify-walk + tree-traverse logic is correct | ✅ stays equal |

## 2. Fix — exact code changes

Make `MedusaWorker` build the verify metadata **using the same kernel + USE_FULL_MASK expansion that NgramWorker uses** (so the two paths are byte-comparable except for draft-token content).

### File: medusa_worker.py

**Change 1 — `__init__`: bump `draft_token_num` to `num_heads + 1`.**

```python
# BEFORE
self.num_heads: int = int(server_args.speculative_num_medusa_heads)
self.draft_token_num: int = self.num_heads          # = 1
assert self.num_heads == 1, "Stage 3b supports only K=1"

# AFTER
self.num_heads: int = int(server_args.speculative_num_medusa_heads)
assert self.num_heads == 1, "Stage 3b supports only K=1"
# Chain length = base accepted token + num_heads draft predictions.
# Matches NgramWorker semantics and --speculative-num-draft-tokens (=2 in prepare_env.sh).
self.draft_token_num: int = self.num_heads + 1      # = 2
```

This already ripples through `MedusaHeads` (still 1 head module), `_use_trained_heads` logic (unchanged), and the K=1 forward path. Only the verify-tree construction below needs to change.

**Change 2 — `_forward_verify_k1`: replace hand-rolled positions/retrive/mask with the same kernel call NgramWorker uses.**

Add the kernel import at the top of the file:
```python
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
```

Replace the entire metadata-construction block (current lines that produce `positions`, `retrive_index`, `retrive_next_token`, `retrive_next_sibling`, `tree_mask`, and `draft_tokens`) with:

```python
bs = batch.batch_size()
ndt = self.draft_token_num                                # = 2

# --- 1. Draft tokens: [base_i, head_pred_i] per req, flat (bs*ndt,) ---
draft_token_list = []
for req in batch.reqs:
    base = req.output_ids[-1]
    if self._use_trained_heads:
        head_pred = getattr(req, "_medusa_draft_token", None)
        if head_pred is None:
            # First verify step before any head has run; pad with 0
            # (mirror NgramCache miss semantics, never reached after step 1).
            head_pred = 0
    else:
        # Stage 3a fallback: no trained heads, repeat base so verify is
        # equivalent to a no-op draft slot (always-reject expected -> bonus=0).
        # Using 0 here keeps it shape-equal to NGRAM cache-miss padding.
        head_pred = 0
    draft_token_list.extend([base, head_pred])

draft_tokens_t = torch.tensor(
    draft_token_list, dtype=torch.int64, device=self.device
)  # (bs*ndt,)

# --- 2. Tree mask: lower-triangular (ndt, ndt) per req, flat (bs*ndt*ndt,) ---
#     Linear K=1 chain: token i can attend to tokens 0..i in the draft window.
tri = torch.tril(torch.ones((ndt, ndt), dtype=torch.bool, device=self.device))
tree_mask_np_torch = tri.unsqueeze(0).expand(bs, ndt, ndt).contiguous()  # (bs, ndt, ndt)
tree_mask_flat = tree_mask_np_torch.reshape(-1)                          # (bs*ndt*ndt,)

# --- 3. Allocate output tensors for the kernel ---
positions = torch.empty(bs * ndt, dtype=torch.int64, device=self.device)
retrive_index = torch.empty(bs, ndt, dtype=torch.int64, device=self.device)
retrive_next_token = torch.empty(bs, ndt, dtype=torch.int64, device=self.device)
retrive_next_sibling = torch.empty(bs, ndt, dtype=torch.int64, device=self.device)

# --- 4. Run the same kernel NGRAM uses to derive positions + retrive_* ---
reconstruct_indices_from_tree_mask(
    tree_mask_flat,
    batch.seq_lens,
    positions,            # out: (bs*ndt,)  e.g. [seq_len, seq_len+1, ...]
    retrive_index,        # out: (bs, ndt)  [[0,1,...]]
    retrive_next_token,   # out: (bs, ndt)  [[1, -1]]
    retrive_next_sibling, # out: (bs, ndt)  [[-1, -1]]
    bs,
    ndt,
)

# --- 5. USE_FULL_MASK expansion (matches NgramWorker for verifier kernel) ---
#     Per-req mask: ones((ndt, seq_len)) concatenated with tri((ndt, ndt)),
#     then flattened.  Shape (sum_i (seq_len_i + ndt) * ndt,).
full_mask_pieces = []
for i, req in enumerate(batch.reqs):
    seq_len_i = len(req.origin_input_ids) + len(req.output_ids)
    left = torch.ones((ndt, seq_len_i - 1), dtype=torch.bool, device=self.device)
    right = tri  # (ndt, ndt)
    full_mask_pieces.append(torch.cat((left, right), dim=1).reshape(-1))
custom_mask = torch.cat(full_mask_pieces, dim=0)

# --- 6. Build NgramVerifyInput with the correct ndt ---
spec_info = NgramVerifyInput(
    draft_token=draft_tokens_t,
    tree_mask=custom_mask,
    positions=positions,
    retrive_index=retrive_index,
    retrive_next_token=retrive_next_token,
    retrive_next_sibling=retrive_next_sibling,
    draft_token_num=ndt,        # ← was hard-coded 1; now 2
)
if self._use_trained_heads:
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
```

The rest of `_forward_verify_k1` (preflight dumps, target forward, hidden capture, verify walk, post-verify req_to_token zeroing, head forward & next-draft cache) is **unchanged**.

**Change 3 — head-forward indexing (small but important).**

After ndt=2, `logits_output.hidden_states` will have shape `(bs * ndt, hidden_size) = (bs * 2, hidden_size)`. The trained-head forward must pick the hidden state at the **last accepted position** per request (not just the row `i`). The simplest correct choice for K=1 is "the hidden state at position 0 of each chain" because that's the one corresponding to the verified base token whose `next` we want the head to predict:

```python
# BEFORE
raw_hidden_states = logits_output.hidden_states            # (bs, H)

# AFTER
# Hidden states come back as (bs * ndt, H).  For K=1 we want one h per req:
# row 2*i (position 0 of chain i) is the hidden state right after the base
# token — the same one a non-spec decode would have produced.
hs = logits_output.hidden_states
if hs is not None and hs.shape[0] == bs * ndt:
    raw_hidden_states = hs.view(bs, ndt, -1)[:, 0, :].contiguous()  # (bs, H)
else:
    raw_hidden_states = hs   # fallback: keep old behaviour
```

This makes the trained-head input identical in shape and semantics to before, **so accuracy is not changed by ndt=2 in the head-forward step**. Where accuracy can change is the verify-walk: with a real second draft now being scored, requests can earn 1 bonus token per step when the head's prediction equals the target's top-1.

### File: prepare_env.sh

**No change required.** It already passes `--speculative-num-draft-tokens $((H+1))` (= 2 for `H=1`); this value was being ignored by MedusaWorker before. Iter-2 makes the worker honor it.

## 3. Why this is safe (compliance + correctness)

- Mirrors **byte-for-byte** the verified-good Ngram metadata path (same kernel call, same `USE_FULL_MASK` expansion, same `NgramVerifyInput` semantics). After iter-2, preflight diff should show **0 critical diffs** in `pre_verify` (only the draft-token *content* differs: ngram gets a cache lookup, medusa gets `0` or the trained head prediction; that diff is **expected and acceptable** — different algorithms).
- The post-verify outputs were already equal in iter-1, proving the verify-walk on `NgramVerifyInput` is correct. We are **not** introducing a new verifier path — we're feeding the existing one with the same-shape inputs ngram gives it.
- Stage 3a fallback (no trained heads) becomes a **proper "always reject" K=1 chain** with a 0-padded draft slot; expected `accept_length = 0` per step, identical wall-clock to today's Stage 3a *except* for one extra forward column — i.e., a tiny regression we accept because Stage 3a is not the shipping config.
- Stage 3b (trained heads) is the **target config**: head-1 predictions are actually verified, expected `accept_length ∈ {0, 1}` per step.
- No rule violations: still single-model, no prefix-cache trick, ndt=2 ≤ K=2 which Stage 3a Medusa codepath already supports via `prepare_env.sh`.

## 4. Validation plan

1. Apply the 3 code changes, commit `medusa: preflight iter 2 — adopt ndt=2 + kernel-built retrive_*`.
2. Push to `minicpm-src`. Start fcloud. Sync.
3. Re-run both modes:
   ```bash
   bash ./preflight_drive.sh ngram     # baseline reference
   bash ./preflight_drive.sh medusa
   ```
4. Pull dumps, run diff. **Pass criterion**: every field except `spec_draft_token` (values only, not shape) and the 4 already-equal post-verify fields should be EQUAL between the two paths. Specifically expect:
   - `draft_token_num`: 2 = 2 ✅
   - `input_ids`, `out_cache_loc`, `spec_positions`, `spec_retrive_*`, `spec_custom_mask`: shapes match, **values match** (positions identical because both call the same kernel on the same `seq_lens`)
   - `seq_lens`: equal once both consume 2 tokens per verify
   - `spec_draft_token`: shape (2,) on both, **values differ** (ngram = lookup, medusa = `[base, 0]` Stage 3a or `[base, head_pred]` Stage 3b) → mark as **expected algorithmic diff**, not a bug.
   - `post_verify`: still all equal in Stage 3a (head_pred=0 will never match the target → 0 bonus, just like ngram cache miss).
5. Document outcome in §5.2 (EN+ZH).
6. Pause fcloud.
7. If §5.2 is clean → **next step is full accuracy + speed eval** to verify Stage 3a regression is ≤ noise and Stage 3b (once trained heads provided) gives the expected speedup.

## 5. Risk

- **R1**: `len(req.origin_input_ids) + len(req.output_ids)` may differ from `batch.seq_lens[i]` at verify time (chunked prefill state). NgramWorker uses the same formula and works, so we inherit its correctness. If the preflight dump shows a `custom_mask` size mismatch, we switch to `int(batch.seq_lens_cpu[i])` as the length source.
- **R2**: `reconstruct_indices_from_tree_mask` expects the tree mask to encode reachability. The lower-triangular `(ndt, ndt)` block is exactly what NGRAM produces for a linear chain (its `mask` numpy from `ngram_cache.batch_get` is lower-triangular for a no-match fallback). Verified by the iter-1 ngram dump showing `retrive_next_token = [1, -1]` = the standard linear chain.
- **R3**: Stage 3b head-forward indexing change (`view(bs, ndt, H)[:,0,:]`) — if instead the verify step relies on **the post-verify accepted hidden** (different row depending on `accept_length`), Stage 3b accuracy could regress when head prediction is correct. Mitigation: keep this index choice (`[:,0,:]`) for iter-2 because Stage 3a will never accept the head's draft, so this index is identical to today's behavior; defer Stage 3b accuracy testing to iter-3 once iter-2 diff is clean.

## 6. Rollback

Single commit — `git revert <iter2-sha>` brings back today's Stage 3a behavior verbatim.

---

**Ask**: approve this iter-2 plan? On `go` I will: apply the 3 edits in medusa_worker.py, commit + push, start fcloud, run both preflight modes, diff, write §5.2, pause fcloud — same loop as iter-1.
