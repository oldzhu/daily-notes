Excellent idea. Let me do thorough research on whether we can apply Marlin's LOP3-based register dequant technique to create a fused INT4→FP8 kernel that actually achieves near-2× speedup over Marlin.

Starting: *Research LOP3 INT4→FP8 feasibility* (1/4)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/dequant.h#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/dequant.h#100-100), lines 100 to 200

Now let me understand the FP8 MMA instruction format requirements more precisely:

Research FP8 MMA format

Searched for files matching `**/sgl-kernel/csrc/gemm/**/*.{cu,h,hpp}`, 33 matches

Searched for regex `m16n8k32.*f32.*e4m3|mma\.sync.*fp8|mfma.*fp8` (`**/sgl-kernel/**/*.{cu,h,hpp,cc}`), 2 results

Searched for regex `ldmatrix|__ldmatrix` (`**/sgl-kernel/**/*.cu`), 11 results

Searched for regex `a_regs|b_regs|cp\[|register.*layout|tensor.*core` (`**/sgl-kernel/**/*.cu`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu)

Searched for regex `SM120|sm120|sm_120|tcgen05` (`**/sgl-kernel/csrc/gemm/**/*.cu`), 20 results

Searched for regex `SM120|tensor.*core|mma\.sync|register.*format` (`**/docs/**/*.md`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#150-150), lines 150 to 180

Searched for regex `fp8.*ldmatrix|ldmatrix.*fp8|tensor.*core.*register` (`**/sgl-kernel/csrc/gemm/**/*.cu`), no results

Searched for regex `mfma|matrix.*fused|mxfp8` (`**/sgl-kernel/csrc/gemm/**/*.{cu,h}`), no results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md)

Searched for regex `fragment|float_e4m3|cutlass.*e4m3` (`**/sgl-kernel/**/*.cu`), 20 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_003.en.md)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu)

Starting: *Design kernel architecture* (3/4)

Now let me research the critical question: can LOP3 be used to convert INT4 directly to FP8 (e4m3) format, similar to how Marlin converts INT4 to FP16/BF16?

Research INT4 to FP8 conversion

Searched for regex `INT4|int4|dequant|fp8|e4m3|QMMA`, 20 results

Searched for files matching `**/sgl-kernel/csrc/gemm/**/*dequant*`, 2 matches

Searched for files matching `**/sgl-kernel/csrc/**/*fp8*`, 20 matches

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/dequant.h#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_dequant.cu#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/dequant.h#150-150), lines 150 to 300

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#1-1), lines 1 to 100

Searched for regex `LOP3|lop3|dequant.*fp8|fp8.*dequant` (`**/sgl-kernel/csrc/gemm/**`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/dequant.h#230-230), lines 230 to 250

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu#1-1), lines 1 to 150

Searched for regex `FE4M3|fp8.*dequant|e4m3` (`**/sgl-kernel/csrc/gemm/marlin/dequant.h`), 12 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/dequant.h#290-290), lines 290 to 350

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_dtypes.cuh#1-1), lines 1 to 50

Searched for regex `bf16.*fp8|fp8.*bf16|convert.*fp8|cvt.*fp8` (`**/sgl-kernel/**`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_mma.cu#1-1), lines 1 to 150

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/int4_to_fp8_lop3_research.md)

This is extremely insightful. Let me now check the actual `__nv_cvt_bfloat16raw_to_fp8` intrinsic and how Marlin's `ldmatrix` works, to finalize the kernel design:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/elementwise/cast.cu#1-1), lines 1 to 60

Now let me also check the CUTLASS-based FP8 GEMM for reference on how a well-optimized FP8 kernel should look:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu#1-1), lines 1 to 60

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu#60-60), lines 60 to 160

Now I have all the information needed. Let me write the comprehensive proposal report.

Starting: *Write proposal report* (4/4)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_optimized_fused_w4a8_v2.en.md)

Completed: *Write proposal report* (4/4)

The full proposal is saved at PROPOSAL_optimized_fused_w4a8_v2.en.md. Here's the executive summary:

---

## Key Findings

### 1. Direct INT4→FP8 LOP3 is **impossible**
FP8 e4m3 has only 3 mantissa bits — can't hold a 4-bit INT4 value. Marlin's trick works because FP16 has 10 mantissa bits.

### 2. Proposed solution: Register-only INT4→BF16→FP8 pipeline
Two-step approach that stays entirely in registers:
- **Step 1**: INT4→BF16 via LOP3 (same technique as Marlin, ~1 instruction)
- **Step 2**: BF16→FP8 via `__nv_cvt_bfloat16raw_to_fp8` (~2 instructions)

This eliminates the **shared memory round-trip** that's killing the current kernel.

### 3. Expected speedup: **2-3× over current fused, 1.2-1.6× over Marlin**
The new kernel should achieve **100-130 TFLOPS** (vs current 35-50, vs Marlin 80-179).

### 4. Two important caveats

**Accuracy risk**: The INT4→BF16→FP8 path loses 3 extra mantissa bits vs the current INT4→FP32→FP8 path. The current kernel already has broken MCQ (26.67%), so the accuracy issue may be fundamental to FP8 MMA precision, not the dequant path.

**Existing CUTLASS QMMA path**: w4a8_fp8_qmma_sm120.cu already exists using NVIDIA's CUTLASS SM100 API — this may already be well-optimized and worth benchmarking first.

### 5. Recommended next step
**Phase 1 first**: Write a small standalone test comparing INT4→BF16→FP8 vs INT4→FP32→FP8 numerical accuracy. If accuracy is acceptable, proceed to full kernel. If MCQ is still broken, the FP8 MMA path itself is the problem and we should abandon this approach entirely.

Would you like me to proceed with Phase 1 (the accuracy prototype), or would you like to discuss the proposal first?

