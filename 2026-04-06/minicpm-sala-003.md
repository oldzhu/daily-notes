

Great question — it's critical to understand causality before committing to a fix. Let me trace the execution timeline to check whether Bug 4 is truly the crash root cause or if FP8 introduces a different issue. 

Searched for regex `attention_kernel_type|self\.attention_kernel_type\s*=` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#300-300), lines 300 to 360

Searched for regex `def create_attention_kernel|class FlashAttn` (`**/python/sglang/srt/layers/attention/minicpm_attention_kernels.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#510-510), lines 510 to 532

Now let me check: when `--force-dense-minicpm` is used, does it switch the backend entirely away from `minicpm_flashinfer`? 

Searched for regex `force.dense.minicpm|force_dense_minicpm` (`**/python/sglang/srt/**`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#1515-1515), lines 1515 to 1530

Let me check whether the FlashAttention kernel path (non-FlashInfer) also has this issue: 

Searched for regex `class FlashAttentionKernel|class FlashInferKernel` (`**/python/sglang/srt/layers/attention/minicpm_attention_kernels.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#118-118), lines 118 to 170

Now let me check what backend was used before FP8 — check when minicpm_flashinfer vs minicpm_flashattn was set: 

Searched for regex `attention-backend|force-dense` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#120-120), lines 120 to 140

Now let me check the effect of `--force-dense-minicpm` to understand the execution path:  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/model_config.py#230-230), lines 230 to 248

Now let me check what happens to the attention backend when `force_dense_minicpm` is True — does it switch to the standard `flashinfer` backend?  

Searched for regex `has_sparse_attention.*backend|minicpm_flashinfer.*backend|minicpm.*backend.*select` (`**/python/sglang/srt/**`), 3 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/attention_registry.py#185-185), lines 185 to 210

Now I have the full picture. Let me trace what happens with `--force-dense-minicpm`:

1. `attention_backend` is `minicpm_flashinfer` → gets changed to `flashinfer` (standard SGLang backend)
2. The model never uses `MiniCPMSparseBackend` at all — it uses the standard `flashinfer` backend

Without `--force-dense-minicpm`:
1. `attention_backend` stays `minicpm_flashinfer` → `MiniCPMSparseBackend` is created
2. Inside it, `attention_kernel_type = "flashinfer"` → `FlashInferKernel` → FlashInfer wrapper CUDA graph path with the buggy kv_indptr code

Now let me check git log to understand when the FlashInfer CUDA graph wrapper code was introduced: 

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline --all -- python/sglang/srt/layers/attention/minicpm_backend.py | head -30

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline 5f3da311c..e08f4bd55 -- python/sglang/srt/layers/attention/minicpm_backend.py

Ran terminal command:  cd /home/oldzhu/sglang && git show --stat 5f3da311c | head -5 && echo "---" && git show --stat e08f4bd55 | head -5 && echo "---" && git show --stat 4bc2850e9 | head -5

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline --all -- python/sglang/srt/layers/attention/minicpm_backend.py | tail -5

Now let me check the crucial commits — what was in our branch before vs what I see now:  

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline HEAD~20..HEAD -- python/sglang/srt/layers/attention/minicpm_backend.py

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline --all | grep -i "CHANGE_0038\|bf16.*bridge\|fp8.*sparse\|sparse.*fp8" | head -10

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline HEAD~40..HEAD | grep -i "0038\|bf16.*bridge\|sparse.*topk\|force.dense"

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline --all | grep -i "force.dense\|force_dense" | head -10

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline de43ca700~3..de43ca700

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline HEAD~60..HEAD | head -40

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline 8109035ba~2..8109035ba

Ran terminal command:  cd /home/oldzhu/sglang && git show --stat 8109035ba | head -20

Now I have the full picture. Here's my answer:

---

**The crash is NOT specifically caused by FP8 or CHANGE_0038.** It's a pre-existing bug in the FlashInfer CUDA graph wrapper for sparse attention.

The git history tells the story clearly:

| Date | Commit | What |
|---|---|---|
| Feb 6 | `5f3da311c` | "enable flashinfer cuda-graph" — first introduced |
| Feb 7 | `e08f4bd55` | "fix flashinfer plan fail bug" — immediate crash fix |
| Feb 7 | `4bc2850e9` | "Add --force-dense-minicpm option" — **workaround added the very next day** |
| Later | `26d7b9ec0` | "[Fix] Fix flash_infer bs (#13)" |
| Later | `e9a1f9ae0` | "[FIX] Fix flashinfer out of range" |

The `--force-dense-minicpm` flag was added **1 day** after FlashInfer CUDA graph support was first enabled, along with multiple subsequent fix attempts. This means the sparse attention + FlashInfer CUDA graph path has **never fully worked**.

**Why it *appears* FP8-specific**: The `--force-dense-minicpm` flag has been in the server args from the beginning. When this flag is active:
- `attention_backend` gets rewritten: `minicpm_flashinfer` → standard `flashinfer`
- `has_sparse_attention` returns `False`
- The entire `MiniCPMSparseBackend` is never instantiated

So the buggy FlashInfer CUDA graph wrapper code was simply **never exercised** — not because FP8 is special, but because `--force-dense-minicpm` bypasses the entire sparse attention codepath regardless of KV cache dtype.

CHANGE_0038 (BF16 bridge for the scorer) fixes a **different, earlier crash** — the scorer rejecting FP8 tensors. It's a prerequisite to even reaching the FlashInfer CUDA graph stage, but it doesn't cause the kv_indptr bug.

**Bottom line**: If you removed `--force-dense-minicpm` with BF16 KV cache (no FP8), you'd hit the same FlashInfer CUDA graph crash. The kv_indptr plan mismatch (Bug 4) is dtype-agnostic.

Does this answer your question? Shall I proceed with the Bug 4 fix?
