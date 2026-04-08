Good — let's evaluate each fix properly before testing.

**Bug 6 (f529418dc) — `cu_seqlens_q_adjusted` → `cu_seqlens_q`**

- **What it fixes**: The `compressed_attention()` function in minicpm_sparse_utils.py calls `infllmv2_attn_stage1` (the sparse scoring kernel during prefill).
- **The original code** used `cu_seqlens_q_adjusted`, which multiplies cumulative sequence lengths by `heads_per_group` (=4). This assumes Q has been reshaped to fold 4 GQA heads into the token dimension (so Q has 4× more rows).
- **The problem**: Q is **not** reshaped that way — it's still `(total_tokens, num_q_heads, head_dim)`. Using the adjusted value makes the kernel think there are 4× more tokens → OOB memory access → crash during **prefill** sparse scoring.
- **Assessment**: This fix is almost certainly correct — it was fixing a real OOB. However, it **only affects prefill**, specifically the non-`--split-stage1` path. It would not affect decode.

**Bug 3 (dbea5948e) — `k_scale` descaling in compression kernels**

- **What it fixes**: `compress_k_to_scratch_kernel` reads keys from the KV cache and mean-pools them to produce compressed keys for sparse scoring.
- **The original code** loaded FP8 keys but didn't multiply by `k_scale` → the compressed keys remained in FP8 quantized scale → scoring comparisons were wrong because Q is in bf16 scale but compressed K was in FP8 scale.
- **Assessment**: Correct for FP8 KV. For bf16 KV, `k_scale=1.0` so the multiply is a no-op — **harmless**.

**Summary table:**

| Fix | Fixes | Affects | Risk of harm | Relevant for bf16 test? |
|-----|-------|---------|-------------|------------------------|
| Bug 8 | int32 overflow in kernel addresses | Decode (large pool indices) | None — pure correctness | Yes, but only if slot index > ~2.1M |
| Bug 6 | OOB in prefill sparse scoring | Prefill only | Very low — restores intended API | Yes |
| Bug 3 | Wrong scale in compressed K scoring | Decode scoring (FP8 only) | None — no-op for bf16 | No (k_scale=1.0) |
| CHANGE_0071 | Triton branch type mismatch | Compile time | None | Only needed because Bug 8 introduced int64 |
| **CHANGE_0070** | kv_indptr "fix" | CUDA graph decode replay | **HIGH — may conflict with captured graph** | **Yes — prime suspect for regression** |

**Key question for Test 3 (pre-0070 baseline):**

The pre-0070 code has none of these fixes. With bf16 KV (6.6M pool), Bug 8's int32 overflow can still bite if a token gets allocated a pool slot > ~2.1M (since `index * 1024` overflows int32). But individual sequences use contiguous blocks that are typically much smaller, so it may work in practice if the allocator gives low-numbered slots.

Bug 6 (cu_seqlens_q OOB) existed pre-0070 too — but it may not crash with bf16 because the memory around Q might be "safely readable" (just wrong values in softmax). It depends on how the memory layout falls.

**Bottom line**: Test 3 should proceed — if it shows >80%, that confirms CHANGE_0070 is the regression source. Then we revert 0070 and keep 6/8/3/71, which should give us the best of both worlds.

Want me to start Test 3 on fcloud?
