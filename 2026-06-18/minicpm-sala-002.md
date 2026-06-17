Analyze cutlass SM90 W4A8 builder

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/collective_mma_array_mixed_input.hpp)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl#1-1), lines 1 to 100

Searched for regex `^}` (`sglang` · `sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl`), 2 results

Searched for regex `^}` (`sglang` · `sgl-kernel/csrc/cutlass_extensions/gemm/collective/collective_mma_array_mixed_input.hpp`), 2 results

Searched for files matching `**/sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/**`, 1 match

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#100-100), lines 100 to 171

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu)

Searched NVIDIA/cutlass for "sm120 mixed_input OR w4a8 OR int4 path:cutlass/gemm/collective", no results

Searched NVIDIA/cutlass for "sm120 OR SM120", 50 results

Searched NVIDIA/cutlass for "Blackwell collective builder", 30 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl#100-100), lines 100 to 150

Searched NVIDIA/cutlass for ""sm120" "mixed_input" OR "CollectiveBuilderMixedInput" path:collective/builders", no results

Searched for files matching `**/sgl-kernel/csrc/cutlass_extensions/gemm/collective/**/*.hpp`, 3 matches

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl#150-150), lines 150 to 276

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp#1-1), lines 1 to 200

Searched for regex `dequant|INT4|int4|scale|zero` (`sglang` · `sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp#200-200), lines 200 to 300

Searched for regex `^}` (`sglang` · `sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu`), 5 results

Searched for regex `^}  // namespace` (`sglang` · `sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp`), 1 result

Searched NVIDIA/cutlass for "CollectiveBuilderMixedInput SM100 OR SM120", no results

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_00_psJ1iDXlCm5GctPWyKep3107__vscode-1779266656963/content.txt)

Here's the detailed comparison based on my analysis of the actual code:

## Option 2: Extend cutlass SM90 W4A8 → SM120

| | |
|---|---|
| **What exists** | `sm90_gmma_builder_mixed_input.inl` (276 lines) + `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp` (**1,536 lines**) |
| **What's needed** | Write SM120 equivalents of both files — new `sm120_*_mixed_input.*` totaling ~1,800 lines of cutlass template code |
| **Key challenge** | SM120 has **56% less SMEM** (101KB vs 232KB on SM90). QMMA block-scale semantics differ from GMMA. Pipeline stage calculation needs full redesign. |
| **NVIDIA precedent** | **None.** NVIDIA cutlass has NO SM120 mixed-input builder. Only SM90 exists. Charting new territory. |
| **Est. effort** | **2–3 weeks** (1 week code, 1–2 weeks debug/tuning) |
| **Pros** | Leverages cutlass TMA/pipeline/epilogue infrastructure; proven for FP8×FP8 SM120 path |
| **Cons** | Extremely complex template metaprogramming; very hard to debug when something breaks; SMEM crunch may force smaller tiles → less speedup |

## Option 3: Raw PTX tcgen05.mma from scratch

| | |
|---|---|
| **What exists** | Our working `w4a8_fp8_fused_gemm.cu` (**170 lines**) — INT4→FP16 dequant + wmma MMA. All the scaffolding (weight unpacking, SMEM layout, tile iteration) already works. |
| **What's needed** | Replace `nvcuda::wmma::mma_sync<m16n16k16>` with `tcgen05.mma` PTX inline assembly. Adapt warp mapping. ~200-300 new lines. |
| **Key challenge** | tcgen05.mma PTX instruction format — need to extract from NVIDIA PTX ISA docs or compiled SASS |
| **Est. effort** | **3–5 days** (1 day PTX research, 1–2 days code, 1–2 days debug) |
| **Pros** | Full control; simple code (~400 lines total); easy to debug; can optimize specifically for our use case; existing kernel scaffolding reuses directly |
| **Cons** | Must implement TMA/epilogue manually (but these are straightforward compared to the MMA itself); risk of suboptimal performance vs cutlass-tuned scheduling |

## My recommendation: **Option 3 (raw PTX)**

The deciding factor is the existing code. We already have a working 170-line kernel that:
- Correctly unpacks INT4 weights in SMEM
- Correctly iterates over tiles
- Correctly writes BF16 output

All that's broken is the MMA instruction itself — `wmma::mma_sync` is slow (warp-level, FP16 accumulator). Replacing it with `tcgen05.mma` PTX (warp-group, FP8 QMMA at 296 TFLOPS) fixes both the correctness bug AND the speed issue in one shot.

Option 2 would require us to throw away all working code and start over in cutlass template land with no NVIDIA reference to follow.
