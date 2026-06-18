Starting: *Research all Path B alternatives exhaustively* (1/3)

Research cutlass SM120 mixed input

Research alternative MMA paths SM120

Searched codebase for "SM120 Blackwell MMA instruction throughput FP8 INT8 tcgen05", 14 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_dequant.cu#1-1), lines 1 to 80

Searched NVIDIA/cutlass for "sm100_umma tcgen05 mma", 6 results

Searched for files matching `**/sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh`, 1 match

Searched NVIDIA/cutlass for "mixed_input sm120 OR blackwell OR sm100", 1 result

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp#1-1), lines 1 to 200

Searched NVIDIA/cutlass for "path:include/cutlass/gemm/collective SM120", 16 results

Searched NVIDIA/cutlass for "tcgen05 int4", 7 results

Searched NVIDIA/cutlass for "sm120_blockscaled_mma", 3 results

Searched NVIDIA/cutlass for "w4a8 mixed_input", no results

Searched NVIDIA/cutlass for "sm120_mma_builder.inl OR sm120_blockscaled_mma_builder.inl", 1 result

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh#1-1), lines 1 to 100

Searched NVIDIA/cutlass for "path:include/cutlass/gemm/collective/builders/ sm120", 7 results

Searched NVIDIA/cutlass for "examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/", no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#100-100), lines 100 to 171

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_dequant.cu#80-80), lines 80 to 200

Searched NVIDIA/cutlass for "tcgen05 Int4 OR int4 conversion scale", 6 results

Searched nvidia/cutlass for "mma_sm100_umma.hpp tcgen05 instruction variants SS TS FP8 F16 BF16 TF32 I8", 64 results

Searched NVIDIA/cutlass for "tcgen05.mma.ws.sync FP8 FP6 FP4 TF32 I8 INT8", no results

Searching 'NVIDIA/cutlass' for 'mma.sync FP8 throughput TFLOPS SM120 Blackwell'



Searched for regex `sm120.*mixed|mixed.*sm120`, 15 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh#50-50), lines 50 to 100

Searching 'NVIDIA/cutlass' for 'include/cutlass/gemm/collective/builders/ path:*.inl sm100'

Fetched https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/builders



Searched for regex `SM100_MMA|SM103_MMA|tcgen05\.mma` (`**/sgl-kernel/csrc/gemm/*.cu`), 12 results

Fetched https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/mma_sm100_umma.hpp

Searching 'NVIDIA/cutlass' for 'SM120 mma instruction FP16 BF16 FP8 INT8 throughput TFLOPS'



Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh#50-50), lines 50 to 150

Fetched https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm

Searching 'NVIDIA/cutlass' for 'sm100_blockscaled_mixed_tma_cpasync_umma_builder.inl'

Searching 'NVIDIA/cutlass' for 'sm100_mixed_input_umma_builder.inl INT4 FP8'

Searched for regex `CollectiveBuilderMixedInput|mixed_input.*builder` (`**/sgl-kernel/csrc/**`), 3 results

Searched nvidia/cutlass for "SM120 mixed input GEMM collective builder blockscaled", 64 results

Searched NVIDIA/cutlass for "sm100_blockscaled_mixed", 1 result



Fetched https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp)

Read memory

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/sm120_mixed_input_gemm_research.md)

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_01_xm1oL74f5or2cl4YxyJV7584__vscode-1779310814934/content.txt)

Searched NVIDIA/cutlass for "mixed_input_gemm int4 fp8 tcgen05 blackwell CuTeDSL", 1 result

Searched nvidia/cutlass for "sm100_mixed_input_umma_builder.inl INT4 FP8 dequant collective builder template", 64 results

Now I have all the research. Let me compile the comprehensive document:

Starting: *Write comprehensive research doc* (2/3)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_W4A8_all_paths_20260521.en.md)

Starting: *Present findings for user review* (3/3)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_W4A8_all_paths_20260521.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/RESEARCH_W4A8_all_paths_20260521.*.md && git commit -m "docs: comprehensive W4A8 all-paths research excluding two-step

Survey of 6 paths for fused INT4->FP8 dequant + GEMM on SM120:
A. SM120 mixed-input cutlass builder (BEST, 2-3 wks)
B. Fix raw PTX kernel (blocked on TMEM/TMA syntax)
C. Port SM100 UMMA mixed-input -> SM120 tcgen05
D. CuTeDSL Python codegen (too complex build)
E. Optimize wmma (caps at 190 TFLOPS)
F. INT8 IMMA (only 136 TFLOPS)

Measured FP8 QMMA: 275-280 TFLOPS vs INT8: 136 TFLOPS." && git push minicpm-src mixed_minicpm_cudagraph 2>&1

Completed: *Present findings for user review* (3/3)

Research complete and committed. Here's the summary:

---

## 6 Paths Evaluated (excluding two-step)

| Path | TFLOPS | Feasibility |
|------|--------|-------------|
| **A. SM120 mixed-input cutlass builder** | ~280 | ✅ **BEST** — create `sm120_mixed_input_mma_builder.inl`, adapt SM90 + SM120 blockscaled patterns |
| B. Fix raw PTX kernel | ~280 | ⚠️ TMEM/TMA syntax hell, unvalidated descriptors |
| C. Port SM100 UMMA → SM120 tcgen05 | ~280 | ✅ Viable fallback — SM100 builder already handles INT4 dequant |
| D. CuTeDSL Python codegen | ~280 | ⚠️ Too complex for our cmake build |
| E. Optimize wmma | ~190 | ❌ Caps at ~190, can't reach 296 |
| F. INT8 dequant + IMMA | ~136 | ❌ Same as BF16 ceiling |

**Key finding**: NVIDIA cutlass has https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/builders/sm100_mixed_input_umma_builder.inl#L0-L21 (~350 lines) that handles all INT4→FP8 dequant logic. But SM120 tcgen05 needs its own builder (different instruction selector). The path is to create `sm120_mixed_input_mma_builder.inl` combining the SM90 mixed-input logic with SM120 blockscaled tcgen05 instruction selection.

Full analysis: `docs/soar_2026_changes/RESEARCH_W4A8_all_paths_20260521.{en,zh}.md`

