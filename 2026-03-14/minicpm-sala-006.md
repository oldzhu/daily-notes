what I mean is use FP8 when FP8 KV cache is enabled, otherwise use BF16 so that it wont corrupt kv cache in gpu global memory, am I wrong?
=========
You are not fully wrong, but that idea only works if the whole compressed-K path is made FP8-aware end to end.

If we say:

KV cache is BF16: compressed K uses BF16
KV cache is FP8: compressed K uses FP8
then that can avoid one specific problem: writing BF16 values into an FP8-backed storage path.

But it still does not solve two other blockers.

The sparse scorer does not accept FP8.
The current top-k scoring path eventually goes into infllmv2_attn_stage1, and that path only supports BF16/FP16. So if compressed K becomes FP8 under FP8 KV mode, the next consumer still fails unless we also add a dequantize/readback bridge before scoring.

The current compression kernel is not doing proper FP8 store semantics.
The normal KV-cache write path is scale-aware in memory_pool.py:995. The compressed-K Triton kernel in minicpm_sparse_kernels.py:217 is not. It accumulates in FP32 and stores straight to key_cache_ptr. So “use FP8 for compressed K when KV cache is FP8” is only safe if we also implement:

explicit quantization
correct scaling
correct FP8 store behavior
correct FP8 read/dequantize behavior for downstream consumers
So the answer is:

conceptually, “match compressed K dtype to KV-cache dtype” is a valid design direction
but in the current codebase, just switching compressed K to FP8 is not enough
you would need a true FP8 compressed-K pipeline, not just a dtype toggle
That is why I have been separating two options:

BF16 scratch compressed-K path
This is the smaller fix.
easiest to stabilize
preserves FP8 KV cache for the main attention path
adds some extra BF16 scratch overhead
True FP8 compressed-K path
This is the cleaner long-term design if it works.
compressed K also stays FP8 when KV cache is FP8
but requires new scale-aware write/read logic and likely changes in the sparse scorer path
higher implementation risk
So your statement is correct only if expanded to:

“Use FP8 compressed K when FP8 KV cache is enabled, with proper FP8 quantize/store/dequantize support for all readers and writers.”

Without that full support, the idea is incomplete.

If you want, I can compare these two implementation routes directly in terms of:

code complexity
expected speed
memory cost
chance of working quickly before the SOAR deadline

