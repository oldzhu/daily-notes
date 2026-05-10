Both excellent catches. Q1 — I was wrong. Q2 — they're independent.

**Q1 — the "max 32K" is incorrect.** During decode, `seq_lens_cpu.max()` is the full prefilled prompt length (32K–128K), not the chunked-prefill chunk size (32K). I confused `--chunked-prefill-size 32768` (a *prefill* chunk cap) with the *decode-time* `max_seq_len`. The actual reduction at our long-context profile is **4× (at 128K) up to 16× (at 32K)** — still huge, but my "16× always" claim was wrong.

**Q2 — the torch.compile crash is a separate bug, unrelated to the fill.** The Round 13d crash was `RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture`, which happens when something inside the captured region calls `torch.cuda.get_rng_state()` / a Generator method. The over-fill is `tensor.fill_(-inf)` — no RNG. So:
  - Fixing the over-fill **does not** unblock `--enable-torch-compile`.
  - If we want torch.compile back on, we'd need a separate change: locate the RNG call inside the sparse-attn region (likely some dropout-related helper or a kernel that grabs RNG even when `p=0`) and either move it outside the captured region or use a deterministic non-Generator path.

Let me correct the doc on both points:  

Replacing 9 lines with 26 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0133_sparse_compress_buffer_oversize.en.md)


```

```


Replacing 9 lines with 24 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0133_sparse_compress_buffer_oversize.zh.md)

Searched for text `16× 浪费` (`**/docs/soar_2026_changes/CHANGE_0133_sparse_compress_buffer_oversize.zh.md`), 1 result

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0133_sparse_compress_buffer_oversize.zh.md#90-90), lines 90 to 110

Replacing 8 lines with 23 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0133_sparse_compress_buffer_oversize.zh.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0133_sparse_compress_buffer_oversize.en.md docs/soar_2026_changes/CHANGE_0133_sparse_compress_buffer_oversize.zh.md && git commit -m "docs(round-13e): correct CHANGE_0133 — 4-16x range and torch.compile is independent

Q1: max_seq_len at decode is full prompt length (32K-128K), not 32K
chunk. --chunked-prefill-size only caps prefill chunks, not decode-time
seq_lens. Reduction is 4x at 128K to 16x at 32K, not always 16x.

Q2: torch.compile RNG/cudagraph-capture crash is unrelated to the
over-fill (fill_(-inf) uses no RNG). Fixing CHANGE_0133 will not by
itself re-enable torch.compile; that requires a separate investigation
of the RNG call site inside the captured region." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Compacted conversationCorrections applied and pushed (`23b6817be`).

**Q1 fix**: Removed the wrong "useful work only touches at most 32K" claim. Decode-time `max_seq_len` is the full prompt (32K–128K) — `--chunked-prefill-size 32768` caps prefill chunks only, not decode `seq_lens`. So real over-fill is **4× at 128K → 16× at 32K**, not "16× always".

**Q2 fix**: Added "Relationship to the Round 13d torch.compile crash (independent issue)" subsection. The torch.compile RNG/cudagraph-capture crash is unrelated to the fill (fill uses no RNG). Dropping `--enable-torch-compile` only worked around RNG/capture incompatibility — it did not "mask" the over-fill. **Fixing CHANGE_0133 alone will NOT re-enable torch.compile in sparse mode**; that's a separate investigation (locate the RNG-touching call site inside the captured region).

Source patch on `minicpm_backend.py:1822-1823` still pending your approval. 



