Correct — these are **speculative-decoding fields**, completely unrelated to sparse-attention topk. Let me show you the exact meaning by reading the Eagle definitions. 

Searched for regex `self\.topk\s*=|self\.draft_token_num\s*=` (`**/python/sglang/srt/speculative/**/*.py`), 20 results

Confirmed. Direct answer:

## What `spec_info.topk` and `spec_info.draft_token_num` mean

Both are **speculative-decoding** fields on the `EagleVerifyInput` object — they describe the **draft-token tree** that the target model has to verify in one forward pass. They have **zero relation** to:
- sparse-attention top-k (the `--dense-as-sparse` / MiniCPM sparse-attn top-k)
- any attention-backend top-k

### `spec_info.topk` (Eagle-only concept)
- Set from `server_args.speculative_eagle_topk` (default 10).
- Means: "at each draft step, how many top-probability tokens to branch on". So each verification batch is a **tree** of shape `topk^num_steps`.
- `topk > 1` → tree-shaped verify → needs custom causal mask (`retrive_next_token`, `retrive_next_sibling`, etc.) to enforce parent-child causality across siblings.
- `topk == 1` → linear K-token verify (a single chain, no siblings) → no tree mask needed.

That's exactly why `hybrid_linear_attn_backend.py:515` writes:
```python
if forward_mode.is_target_verify() and spec_info.topk > 1:
    # populate retrieve_next_token / retrieve_next_sibling for tree-mask attention
```

### `spec_info.draft_token_num`
- Total number of draft tokens to verify per request in this step. For Eagle: `topk^steps` (tree size). For NGRAM: a flat number (e.g. 12). For Medusa: `num_medusa_heads + 1`.
- Used in `_replay_metadata` to build `query_start_loc` for the verify-shape attention (one verify call sees `bs * draft_token_num` query positions).

### Your config
You're 100% right:
- `--attention-backend flashinfer` (stock) → the **flashinfer** backend handles attention, not minicpm/sparse.
- `--force-dense-minicpm` (or stock flashinfer dropping that flag, either way) → **dense full attention**.
- "sparse attention top-k" never comes into play.

The `spec_info.topk` field only became a problem because the hybrid GLA backend (which still handles the **GLA/Mamba layers** of MiniCPM-SALA — the 24 lightning-attn layers, alongside the 8 std-attn layers that go to flashinfer) was written assuming spec_info has Eagle's tree-shaped layout. NGRAM (and Medusa) instead supply a **linear K-token verify** (effective topk=1, no tree), so the field doesn't exist on their spec_info objects.

### So the fix is conceptually trivial

For non-Eagle algorithms, the verify is **just a linear chain**, which means the hybrid backend should take the "else" branch (no tree-mask). The 3-line tolerance patch (`getattr(spec_info, "topk", 1)`) just teaches the hybrid backend that "missing topk attribute" = "linear verify, no tree", which is the truth for NGRAM/Medusa.

**TL;DR — both fields are purely speculative-decoding draft-tree descriptors. Independent of attention backend choice and of sparse-attention top-k. Your understanding is correct.**
