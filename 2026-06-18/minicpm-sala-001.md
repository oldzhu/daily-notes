Find prior W4A8 FP8 tests

Searched for regex `W4A8|FP8 blockwise|fp8_blockwise_scaled_mm|gptq_int4_to_fp8_blockwise|Phase B1` (`**/docs/soar_2026_changes/**`), 100 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#1-1), lines 1 to 200

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/transcripts/bf8d10bb-81df-4c7d-aacb-b69c96e571b1.jsonl#1-1), lines 1 to 100

You're absolutely right. The documents confirm it clearly:

### Prior test: W4A8#1 (two-step, INT4→FP8 at load time) — **NET REGRESSION**

From `TEST_RESULTS_TRACKING.md`:
| Metric | Marlin Baseline | W4A8#1 Two-Step | Delta |
|--------|----------------|-----------------|-------|
| S1 | 121.71s | 265.32s | **+118%** |
| S8 | 44.09s | 68.88s | **+56%** |
| Smax | 35.86s | 46.44s | **+30%** |

Root cause: "FP8 weight storage = 2× weight memory footprint vs Marlin INT4. At decode bs=1 the kernel is memory-bandwidth-bound → 2× slower."

### The true fused approach (W4A8-REAL-v25) — **8-9% SPEEDUP**

| Metric | Marlin Baseline | W4A8-REAL Fused | Delta |
|--------|----------------|-----------------|-------|
| S1 | 121.71s | 110.79s | **−9.0%** |
| S8 | 44.09s | 40.51s | **−8.1%** |
| Smax | 35.86s | 32.67s | **−8.9%** |
| Accuracy | 79.29% | **81.07%** | **+1.78pt** (best ever) |

W4A8-REAL kept INT4 storage + dequantized in-kernel → same memory bandwidth as Marlin, but compute at 296 TFLOPS (QMMA) vs 148 TFLOPS.

### Conclusion

Phase B1 (two-step) is a confirmed dead end — no need to redo it. The path forward is the **true fused kernel** (Phase B2). The question is which approach to take:

1. **Fix/debug the existing MMA fused kernel** (`w4a8_fp8_fused_gemm.cu`) — had multi-tile correctness bug (~68 error) and server 503. But W4A8-REAL-v25 apparently worked — was that using a different kernel path?

2. **Extend SM90 W4A8 cutlass to SM120** — adapt w4a8_grouped_mm_c3x.cuh (already handles INT4→FP8 dequant in cutlass mainloop) to use tcgen05 instead of wgmma

3. **Raw PTX tcgen05.mma** — write from scratch

Do you recall what kernel was actually used for the W4A8-REAL-v25 test that got the 8-9% speedup? That would tell us what's already working and where to focus.
