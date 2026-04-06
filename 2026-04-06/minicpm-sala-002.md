

 

Starting: *Path A: Fix sparse attention crash* (2/3)

Let me do a thorough investigation of the sparse attention code in the correct location (srt): 

Path A: sparse attention crash deep dive

Searched for files matching `**/minicpm_backend.py`, 2 matches

Searched for files matching `**/minicpm_sparse_kernels.py`, 2 matches

Searched for files matching `**/minicpm_sparse_utils.py`, 2 matches

Searched for files matching `**/minicpm_attention_kernels.py`, 2 matches

Searched for files matching `**/mem_cache/common.py`, 2 matches

Ran terminal command:  wc -l /home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py /home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py /home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py /home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py /home/oldzhu/sglang/python/sglang/srt/mem_cache/common.py

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#1-1), lines 1 to 823

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#1-1), lines 1 to 531

Read [](file:///home/oldzhu/sglang/python/sglang/srt/mem_cache/common.py#1-1), lines 1 to 635

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#200-200), lines 200 to 500

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#200-200), lines 200 to 500

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#500-500), lines 500 to 800

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#500-500), lines 500 to 800

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#800-800), lines 800 to 1100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#800-800), lines 800 to 1100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1100-1100), lines 1100 to 1500

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1100-1100), lines 1100 to 1400

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1500-1500), lines 1500 to 1932

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1400-1400), lines 1400 to 1751

Searched for text `sparse_cache_seqlens` (`**/python/sglang/srt/**/*.py`), 20 results

Searched for text `write_sparse_k1` (`**/python/sglang/srt/**/*.py`), 10 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/mem_cache/memory_pool.py#560-560), lines 560 to 580

Read [](file:///home/oldzhu/sglang/python/sglang/srt/mem_cache/memory_pool.py#620-620), lines 620 to 660

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/schedule_batch.py#2040-2040), lines 2040 to 2090

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/schedule_batch.py#2090-2090), lines 2090 to 2170

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_bdrk_011FXmjaYRL8i6PsXcM1Fz8Q__vscode-1775430475944/content.txt)

Good news: Bugs 1 and 2 are already fixed in srt. The two remaining issues are:

- **Bug 4 (CRITICAL)**: kv_indptr plan/overwrite conflict in CUDA graph — this is the crash root cause
- **Bug 3 (QUALITY)**: FP8 values not descaled in scratch compress kernel — quality degradation

Let me now read the exact code for Bug 4 to design a fix:  

Searched for text `convert_sparse_page_table_to_flashinfer` (`**/python/sglang/srt/layers/attention/minicpm_attention_kernels.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#340-340), lines 340 to 430

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#430-430), lines 430 to 480

Now let me check the CUDA graph decode path more carefully — how is `begin_forward` called during replay:   

Searched for text `init_forward_metadata_replay_cuda_graph` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1676-1676), lines 1676 to 1810

Now I can see the critical issue clearly. Let me check what `kv_indptr_view` looks like — the static precomputed one:   

Searched for text `flashinfer_kv_indptr` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 20 results

Compacted conversation

Starting: *Understand kv_indptr full lifecycle* (1/6)

Let me read the key sections of the backend to fully understand the kv_indptr lifecycle — capture, replay, and in-graph execution:   

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1380-1380), lines 1380 to 1500

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1580-1580), lines 1580 to 1810

Now let me see what `placeholder_kv_indptr` is and how `convert_sparse_page_table_to_flashinfer` updates kv_indptr inside the graph:  

Searched for text `placeholder_kv_indptr` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#1-1), lines 1 to 50

Searched for text `def convert_sparse_page_table_to_flashinfer` (`**/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1290-1290), lines 1290 to 1315

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#785-785), lines 785 to 824

Searched for text `def convert_sparse_to_flashinfer_two_kernel` (`**/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#700-700), lines 700 to 785

Now let me read the FlashInferKernel.forward() to see how `convert_sparse_page_table_to_flashinfer` is called in-graph: 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#300-300), lines 300 to 500

Let me also understand the full forward path — how sparse attention scoring happens relative to FlashInfer: 

Searched for regex `def forward_decode|def forward\b` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1111-1111), lines 1111 to 1290

Now let me check how `sparse_cache_seqlens_int32_cpu` is computed to understand whether entries always have exactly `num_sparse_topk_tokens`:   

Searched for regex `sparse_cache_seqlens_int32_cpu` (`**/python/sglang/srt/**`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/schedule_batch.py#2050-2050), lines 2050 to 2080

I see that `sparse_cu_seqlens_k_cpu` is already the correct cumsum. Let me verify it's available in the replay context: 

Searched for text `sparse_cu_seqlens_k_cpu` (`**/python/sglang/srt/model_executor/forward_batch_info.py`), 6 results

Good — `forward_batch.sparse_cu_seqlens_k_cpu` is available. Now let me also check Bug 3 — the FP8 descaling in the scratch kernel:  

Searched for regex `compress_k_to_scratch_kernel|def compress_k_to_scratch` (`**/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#1-1), lines 1 to 130

Good, I now have a complete understanding. Let me also look at how `sparse_get_topk_impl` handles FP8 (Bug 1 fix) to understand the existing bridge pattern: 

Searched for regex `def sparse_get_topk_impl|bf16.*bridge|BF16.*bridge` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 3 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#811-811), lines 811 to 860

Now I have a complete understanding of Bug 4. Let me read the full replay flashinfer section to prepare the exact edit: 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1745-1745), lines 1745 to 1815

Starting: *Design Bug 4 fix approach* (2/6)

I now have a complete understanding. Here's my **optimization proposal** for CHANGE_0070: Fix sparse attention CUDA graph crash (Bug 4: kv_indptr plan mismatch).

---

## Optimization Proposal: CHANGE_0070 — Fix Sparse Attention CUDA Graph Crash

### Objective and Expected Gain
- **Enable sparse attention** by removing `--force-dense-minicpm` from server args
- **Expected speedup: 20-40%** on long-context inputs (35% of eval = 32-128K, 15% = 128-160K)
- Sparse attention attends only to top-K selected tokens instead of full context window

### Root Cause Analysis (Bug 4)

The crash occurs in the FlashInfer CUDA graph replay path due to a **kv_indptr plan mismatch**:

1. **Pre-allocation** (minicpm_backend.py): `placeholder_kv_indptr = arange(0, max_bs*2+1) * num_sparse_topk_tokens` — static pattern `[0, K, 2K, ...]`

2. **Capture** (minicpm_backend.py): `begin_forward()` plans with this static pattern. The captured graph body runs `convert_sparse_page_table_to_flashinfer()` which **overwrites** `kv_indptr_view` with `cumsum(zeros)` = `[0, 0, 0, ...]`

3. **Replay** (minicpm_backend.py): Reads from the **same pre-allocated buffer** — which now contains **stale data** from the previous graph execution (or zeros from capture). Calls `begin_forward()` with these stale values → FlashInfer plans workspace with **wrong partitioning**

4. **Graph execution**: `convert_sparse_page_table_to_flashinfer()` computes the **correct** kv_indptr from current `sparse_cache_seqlens`, but the plan is already wrong → workspace OOB → crash

Additionally, `sparse_cache_seqlens` per entry is **NOT always K** — it varies with sequence length and block alignment (see schedule_batch.py):
```python
sparse_cache_seqlens = where(seq_lens <= K, seq_lens, (topk-1)*block_size + seq_lens%block_size)
```

### Fix

In `init_forward_metadata_replay_cuda_graph()`, **compute correct kv_indptr from `forward_batch.sparse_cu_seqlens_k_cpu`** (already available — it's the padded cumsum of sparse_cache_seqlens) and write it to the pre-allocated buffer BEFORE calling `begin_forward()`.

### Files to Change
1. **minicpm_backend.py** — Replay function: replace static kv_indptr comment and add correct cumsum computation
2. **prepare_env.sh** — Remove `--force-dense-minicpm` to enable sparse attention

### Risk to Accuracy/Stability
- **Low risk**: This is a correctness fix, not a behavioral change. Sparse attention was always intended to work; the dense fallback was a crash workaround.
- **Accuracy**: Should match or improve current accuracy (sparse attention with FP8 cache already validated in non-CUDA-graph mode)
- **We keep `--force-dense-minicpm` available as rollback** if any issues

### Rule Compliance
- No prohibited techniques used
- Correctness preserved (sparse attention is the model's native mode)
- KV cache format unchanged

### Test & Benchmark
```bash
# Test correctness
python benchmark/soar/demo_sala/eval/eval.py --preset sparse_qkv_w8

# Test speed (s1, s8, smax)  
python benchmark/soar/demo_sala/eval/speed_eval.py
```

### Rollback
- Add `--force-dense-minicpm` back to SGLANG_SERVER_ARGS in prepare_env.sh

---

**Do you approve this change?**
