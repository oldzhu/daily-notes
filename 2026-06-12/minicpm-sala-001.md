You're right on both counts — it's a guess, not a sure fix. And the crash-at-startup question is key. Let me think more carefully:

## Why CUDA Graph Capture Crashes at Startup (No Requests)

CUDA graph capture runs a **dummy forward pass** through every layer to record operations. The crash at `hidden_states *= self.residual_scale` is in the attention layer, but the *actual illegal access* likely happens earlier in the W4A8 FP8 GEMM (async CUDA errors are reported late).

The deeper issue: `cutlass_w8a8_block_fp8_linear_with_fallback` internally allocates **temporary FP8 buffers** for activation quantization. During CUDA graph capture, these temp buffers get baked into the graph at fixed addresses. On replay, if the buffer isn't re-allocated before use, you get an illegal memory access. The Marlin INT4 kernel doesn't have this problem — it uses a pre-allocated workspace.

So `.clone()`/`.contiguous()` on the output probably **won't fix it** — the problem is upstream in the temp FP8 buffer lifecycle during capture.

## Better Plan: Systematic Debugging, Not Guessing

Let me test 3 things to narrow down the root cause:

1. **Boot without CUDA graph** (`--disable-cuda-graph`) — confirms if it's capture-only
2. **Boot with `CUDA_LAUNCH_BLOCKING=1`** — pinpoints the exact kernel that illegally accesses memory
3. **If step 1 succeeds**, the fix is in making the FP8 path CUDA-graph-safe (likely pre-allocating temp buffers)

Do you want me to run step 1 now on fcloud (boot without cuda graph)? It's a one-flag change and will tell us definitively if the W4A8 path works at all outside of graph capture.
