Good question. Let me trace through exactly how the standalone `.so` gets loaded and used in the server process:

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#973-973), lines 973 to 1000

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1001-1001), lines 1001 to 1035

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#133-133), lines 133 to 143

Searched for regex `_soar_w4a8_real_active` (`**/python/sglang/srt/layers/quantization/gptq.py`), 3 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1036-1036), lines 1036 to 1050

## Answer: How the standalone `.so` is used in the sglang server

Yes, you're right on both counts:
- **Official sgl-kernel wheel**: still **NOT built** (blocked by FetchContent git clone issues on fcloud)
- **Standalone `.so`**: built separately via minimal CMake, and it **crashed the server**

Here's how the standalone `.so` gets loaded and used in the server process:

### Loading mechanism

The `.so` is dynamically loaded at runtime — it is **not** part of the sgl-kernel wheel. Here's the chain:

1. **Server starts** → loads the GPTQ model → during weight setup, gptq.py saves the original (pre-Marlin-repack) `qweight`/`qzeros`/`scales` and sets `layer._soar_w4a8_real_active = True`.

2. **On first forward pass**, gptq.py's `apply()` is called. It hits this block (line 976-983):
   ```python
   if getattr(layer, "_soar_w4a8_real_active", False):
       _fused_so = "/root/submission_sim/libw4a8_fused_gemm.so"
       if os.path.exists(_fused_so):
           torch.ops.load_library(_fused_so)  # <-- dynamically loads the .so
   ```

3. **`torch.ops.load_library()`** loads the shared library into the **current process** (the sglang server process). The `.so` has a `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)` that registers `w4a8_fp8_fused_gemm` into the same `torch.ops.sgl_kernel` namespace that the main sgl-kernel wheel uses.

4. **The op is now callable** as `torch.ops.sgl_kernel.w4a8_fp8_fused_gemm(...)`.

### When the fused kernel fires (or not)

| Condition | Path taken |
|-----------|-----------|
| `M % 128 == 0` (large prefill) | ✅ Fused kernel called |
| `M % 128 != 0` (decode, M=1) | ❌ Raises RuntimeError → caught → **fallback** |
| Any exception in fallback | Flags `_soar_w4a8_real_active = False` permanently |

### Why the crash

The server crash almost certainly happens when the fused kernel path is hit for the **first time during prefill** (M ≥ 128). The decode-only smoke test (`"What is 2+2?"`, M=1 token) worked because it took the fallback path. The crash is likely caused by:

- **Dimension misalignment** in some model layer (e.g., `out_features` not divisible by 8, or `in_features` not aligned to `group_size` boundaries)
- **Out-of-bounds memory access** in the dequant indexing (`qweight[kp * N + ng]` or `qzeros[gid * (N/8) + zn]`) when N or K isn't perfectly aligned
- **Grid launch with partial tiles** producing incorrect results that cascade into a later crash

To debug, we need to restart the server with `CUDA_LAUNCH_BLOCKING=1` and capture the actual error. That will tell us exactly which layer dimension triggers the bug. Ready to debug?
