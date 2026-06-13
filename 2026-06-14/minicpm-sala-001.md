This is a major kernel development task. Let me start by understanding both existing kernels to design the fusion:

Explore W4A8 CUDA kernels

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_dequant.cu#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#1-1), lines 1 to 200

Searched for text `fp8_blockwise_gemm|w4a8_fp8_dequant` (`**/sgl-kernel/CMakeLists.txt`), 2 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#200-200), lines 200 to 400

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#400-400), lines 400 to 466

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin.cuh)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/kernel.h)

Searched for text `GEMM_SOURCES|gemm_sources` (`**/sgl-kernel/CMakeLists.txt`), 2 results

Searched for regex `__global__|torch::Tensor.*gemm|std::tuple` (`**/sgl-kernel/csrc/gemm`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_dequant.cu#160-160), lines 160 to 216

Searched for text `__global__` (`**/sgl-kernel/csrc/gemm/qserve*`), 4 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#285-285), lines 285 to 330

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu#1-1), lines 1 to 80

Searched for text `__global__ Marlin` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu#320-320), lines 320 to 340

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1-1), lines 1 to 100

Searched for text `w4a8_group_gemm|torch::Tensor` (`**/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu`), 8 results

Searched for text `__global__|torch::Tensor` (`**/sgl-kernel/csrc/gemm/per_tensor_quant_fp8.cu`), 5 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu#1-1), lines 1 to 180

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu#665-665), lines 665 to 780

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/per_tensor_quant_fp8.cu#45-45), lines 45 to 124

Searched for text `fp8_blockwise|dispatch` (`**/sgl-kernel/csrc/gemm/cutlass_extensions`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#1-1), lines 1 to 80

Searched for files matching `**/gemm/**/utils.h`, no matches

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc)

Searched for text `pad_tensor|getSMVersion|dispatch` (`**/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`), 13 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions)

Searched for text `fp8_blockwise` (`**/sgl-kernel/csrc/cutlass_extensions/gemm`), no results

Searched for regex `getSMVersion|pad_tensor` (`**/sgl-kernel/csrc/**/*`), 16 results

Searched for regex `getSMVersion|pad_tensor` (`**/sgl-kernel/csrc/**/*.h`), no results

Searched for regex `getSMVersion|pad_tensor` (`**/sgl-kernel/csrc/**/*.cuh`), no results

Searched codebase for "getSMVersion getSMArch", 25 results

Searched for text `pad_tensor` (`**/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`), 2 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/include/utils.h#1-1), lines 1 to 400

Searched for regex `pad_tensor|dispatch.*sm90|dispatch.*sm120` (`**/sgl-kernel/csrc/gemm/cutlass_extensions`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm)

Searched for text `pad_tensor` (`**/sgl-kernel/csrc/cutlass_extensions/gemm/*.cuh`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/cutlass_gemm_caller.cuh)

Searched for regex `torch::Tensor pad_tensor` (`**/sgl-kernel/**/*.cu`), no results

Searched for text `pad_tensor` (`**/sgl-kernel/csrc/common_extension.cc`), no results

Searched for regex `pad_tensor`, 30 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/include/utils.h#430-430), lines 430 to 470

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_00_4T8jCLvPmcqbOOKaruqE1430__vscode-1779160981845/content.txt#1-1), lines 1 to 200

Now I have the full picture. The proper fix is a **fused kernel** that reads INT4 weights and dequants them inside the GEMM. Let me write a focused proposal before coding:

---

## Fused INT4→FP8 GEMM Kernel — Design Proposal

### Problem
Current W4A8 stores FP8 weights (8-bit) in HBM. The FP8 GEMM reads 8-bit weights vs Marlin reading 4-bit weights → **2× bandwidth penalty** → 28-118% slower.

### Solution Architecture
Instead of dequantizing all weights upfront, **fuse dequant into the GEMM tile loop**:

```
For each 128×128 weight tile:
  1. Load INT4 qweight tile (4-bit → 64 bytes)
  2. Unpack INT4 → FP8 in SMEM
  3. Load FP8 activation tile  
  4. Run FP8 QMMA (296 TFLOPS)
  5. Accumulate to output
```

### Files to Create/Modify
| File | Change | Purpose |
|------|--------|---------|
| `sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu` | **NEW** | Fused kernel: INT4 dequant + FP8 MMA |
| w4a8_fp8_dequant.cu | Modify | Add inline `__device__` dequant function for reuse |
| CMakeLists.txt | Modify | Add new .cu file to build |
| gptq.py | Modify | Call fused kernel instead of separate dequant+GEMM |

### Reuse Strategy
Instead of writing a full cutlass GEMM from scratch, I'll **adapt the existing QServe W4A8 kernel** (`qserve_w4a8_per_group_gemm.cu`) which already does INT4 weight loading + int8 compute. The change: replace int8 MMA with FP8 QMMA. This is simpler than starting from scratch.

### Expected Gain
- Weight HBM traffic: 8-bit → 4-bit = **2× reduction**
- S1 expected: 265s → ~140s (close to Marlin 122s)
- S8/Smax: FP8 QMMA at 296 TFLOPS should outperform Marlin at 74 TFLOPS when compute-bound

### Risk & Timeline
- **Risk**: Medium — new kernel, needs correctness validation
- **Timeline**: 2-3 days (kernel write + build + test)

---

Do you want me to proceed with this approach, or would you prefer a simpler intermediate step first?
