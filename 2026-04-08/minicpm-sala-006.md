You raise an excellent point. Let me trace through the logic:

**Was CHANGE_0070's crash diagnosis correct?**

The CUDA graph crash happened when sparse decode ran with FP8 KV. CHANGE_0070 attributed it to `kv_indptr` mismatch. But Bug 8 is equally compelling as the actual cause:

- FP8 KV pool = ~16.2M tokens
- Sparse kernels compute: `token_k_indices * head_num_k * head_dim = index * 8 * 128 = index * 1024`
- int32 max = ~2.1 billion → overflows when `index > 2.1M`
- With a 16.2M pool, **~87% of token slots** produce corrupted addresses → illegal memory access → crash

Even with bf16 KV (6.6M tokens), overflow starts at index 2.1M, affecting ~68% of the pool. The pre-0070 code may have "worked" for sequences that happened to get low-index slots, and produced silent corruption for others.

**Why CHANGE_0070 might be harmful:**

Before 0070, `kv_indptr = [0, K, 2K, ...]` is EXACTLY what `convert_sparse_page_table_to_flashinfer()` produces during CUDA graph *capture* (because capture sets every `sparse_cache_seqlens = K`). So the static pattern was **consistent** with the captured graph's workspace plan.

After 0070, `kv_indptr` is set to dynamic values from `sparse_cu_seqlens_k_cpu` in the *replay* step, then `begin_forward()` replans the workspace. But the *captured* graph was planned with the static pattern. If FlashInfer's workspace layout depends on sequence lengths, the replay plan may conflict with what the captured graph expects → corrupted attention output → accuracy drop.

**Assessment: You are very likely right.** The crash was probably Bug 8, and CHANGE_0070's kv_indptr rewrite may have introduced the accuracy regression from >80% to ~57%. Test 3 (pre-0070 baseline) will confirm this.

Shall I proceed with running Test 3 on fcloud?
