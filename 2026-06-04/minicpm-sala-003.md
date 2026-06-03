Searched for regex `lightning|linear_attn|minicpm.*attention|hybrid` (`**/python/sglang/srt/models/minicpm*.py`), 40 results

Searched for files matching `**/minicpm*sala*.py`, no matches

Searched for regex `forward_extend|forward_decode|forward_target_verify|SimpleGLAAttnBackend|verify` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#480-480), lines 480 to 540

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#490-490), lines 490 to 600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#970-970), lines 970 to 1080

Two honest answers:

## 1. Pre-flight on local without NVIDIA GPU — **no, not feasible**

sglang is GPU-only end-to-end. Everything the pre-flight needs to compare is CUDA:

- **Model forward**: weights live in GPU memory; flashinfer prefill/decode kernels are CUDA.
- **Lightning attention path** (MiniCPMLightningMixer): uses `causal_conv1d_update`, `chunk_gla` — both CUDA kernels with no CPU fallback.
- **Verify kernel** (`verify_tree_greedy`) and the tree-mask reconstruction kernel: CUDA only, in sgl-kernel.
- **KV cache / req_to_token pools**: GPU tensors, no host-side mirror.

A "stub everything with torch CPU" version would diverge so much from the real path that any byte-by-byte comparison loses meaning. The pre-flight only adds signal if it exercises the **same** code path the failed fcloud run did.

**Practical options:**

- **(a)** Run the pre-flight as a short interactive Python script **on fcloud** itself. Cost: ~5–10 min of GPU time per iteration vs ~2 h for a full eval. That's the option I'd recommend — still 10–20× cheaper than another full submission attempt.
- **(b)** Rent a cheap SM≥80 GPU machine elsewhere just for development (e.g., a colab/A10/4090 spot). MiniCPM-SALA loads in ~30GB so a 40GB+ card is enough for K=1. But this is logistical overhead.
- **(c)** Skip pre-flight, do full fcloud eval. That's the coin-flip I flagged.

So "pre-flight" is not necessarily local — it's "single-process scratch script, not the full eval harness". It still costs fcloud GPU minutes.

## 2. Lightning attention + verify — already customized, but only for **linear-chain** (topk=1)

Yes, this customization exists. I checked the source. Two things are true and one is uncertain:

**True ①** — The hybrid linear-attn backend has explicit `is_target_verify` branches:

- hybrid_linear_attn_backend.py: `_capture_metadata` checks `if forward_mode.is_target_verify() and getattr(spec_info, "topk", 1) > 1:` to decide whether to use the Eagle tree retrieve_* fields or skip them.
- The fallback ("topk missing or ==1") is the **linear-chain verify path**, which uses only `custom_mask` and `query_start_loc`. This is exactly the case for **Medusa K=1 / ndt=2** and Ngram.

**True ②** — `forward_extend` (around hybrid_linear_attn_backend.py) has a complete verify path that:
- Reshapes mixed_qkv to `(batch, draft_token_num, -1)`
- Calls `causal_conv1d_update` with `intermediate_conv_window` / `intermediate_state_indices` so each verify position gets the right pre-state
- Branches into the GLA SSM kernel with tree-aware state management

That branch is fully written for ndt > 1. The code comments and the prior SOAR work (PROPOSAL_medusa_stage3 §12) explicitly note: "NGRAM/Medusa attention only needs custom_mask; retrieve_* fields are unused on this backend for those algos." So the linear-attn side **looks correct** for K=1.

**Uncertain ③** — There's an `assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)` inside the verify path. The `SpeculativeState` mamba pool is allocated only when the runtime knows speculative decoding is on. **I haven't verified that MedusaWorker triggers the same allocation hooks that the Eagle/Ngram workers trigger.** If it doesn't, the assert fires on first decode step. But the Stage 3a logs we have don't show that assertion firing — they show normal decoding with the wrong KV content. So the pool allocation is probably fine; this is a "still check during pre-flight" rather than "known bug".

## Bottom line on (2) and its relation to the catastrophic failure

- The lightning-attn verify customization **is already done** and looks complete for the K=1 case we need.
- So lightning attention is **probably not** the cause of the ndt=2 catastrophe. The cause is more likely on the **full-attention layers** (flashinfer/RadixAttention) where my positional-invariant hypothesis lives: the bonus position's KV gets written with `output_ids[-1]`'s content, the head's draft prediction reads from that poisoned slot, and the resulting target_predict drifts.
- But **uncertainty ③** means the pre-flight script must include a "did SpeculativeState mamba pool get allocated and used correctly under MedusaWorker?" check, not just a full-attention KV inspection. That's an additional check item.

## Updated recommendation

If you decide to proceed:

1. Build the pre-flight as a single Python script that runs **on fcloud** (10 min job, not a 2 h eval). Have it dump positions, retrive_index, tree_mask, KV-slot contents (full attn), and SpeculativeState contents (lightning attn) for one MedusaWorker step and one NgramWorker step side-by-side. Diff them.
2. Iterate fixes locally based on the diff. Only re-run the pre-flight on fcloud, not the full eval, until the diff is clean.
3. Then and only then run the full accuracy eval.

If you'd rather skip pre-flight and bet on the positional-invariant fix as-is, I can do that, but understand we'd be using fcloud accuracy eval as the debugger — expensive feedback loop. 


