Let me trace the dense vs sparse paths in the backend. 

Searched for regex `force_dense|dense_forward|forward_extend|forward_decode|compressed_attention|topk_idx` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#906-906), lines 906 to 1000

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1000-1000), lines 1000 to 1120

Now let me check the decode path and how dense is handled:  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1117-1117), lines 1117 to 1260

Now let me check what `--force-dense-minicpm` does — does it use a completely different backend? 

Searched for text `force_dense` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), no results

Searched for regex `force_dense_minicpm|force.dense.minicpm`, 20 results

Searched for regex `force_dense_minicpm.*backend|minicpm_flashinfer.*force_dense` (`**/python/sglang/**"`), no results

Searched for regex `force_dense_minicpm` (`**/python/sglang/**/*.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#1518-1518), lines 1518 to 1530

Now I have the complete picture. Here's the answer:

**No, top-k is NOT used in dense mode.** They are completely different backends.

When `--force-dense-minicpm` is set:
- The attention backend is switched from `minicpm_flashinfer` → standard `flashinfer` (or `minicpm_flashattn` → `fa3`)
- `has_sparse_attention` returns `False`, `sparse_layer_ids` returns `[]`
- The `MiniCPMSparseBackend` class is never instantiated
- Standard FlashAttention/FlashInfer is used — every Q attends to **all** K/V tokens with no scoring or selection step

**Dense vs Sparse comparison:**

| Aspect | Dense (`--force-dense-minicpm`) | Sparse (default) |
|--------|------|--------|
| Backend | Standard `flashinfer` / `fa3` | `MiniCPMSparseBackend` |
| K compression | None | Average 32 K vectors → 1 compressed K |
| Score computation | None (attend to everything) | `Q @ K_compressed^T` on ~124 compressed vectors |
| Block selection | None | `topk(9)` from ~31 block scores |
| Final attention | Q × all K/V (full sequence) | Q × selected 576 tokens only |
| Complexity | O(seq_len) per query | O(compressed_len) scoring + O(topk × block_size) attention |

Dense mode is simply standard full attention — no scoring, no selection, no compression. The entire top-k mechanism only exists in the sparse path.
