Created 3 todos

Let me do a thorough investigation of the full history, change docs, and related materials.

Research CUTLASS QMMA history

Searched for files matching `**/docs/soar_2026_changes/**/*`, 200 matches

Read [](file:///home/oldzhu/sglang/3rdparty)

Read [](file:///home/oldzhu/sglang/sgl-kernel)

Searched for regex `QMMA|sm120|w4a8|fused|CUTLASS|sm100|W4A8_REAL` (`**/docs/soar_2026_changes/**`), 100 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_QMMA_TCGEN05.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4A8_REAL_001.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_REAL_FP8_GEMM.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md)

Searched for files matching `**/sgl-kernel/csrc/gemm/w4a8*`, 6 matches

Searched for regex `cutlass|CUTLASS` (`**/sgl-kernel/**`), 50 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_002.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_W4A8_all_paths_20260521.zh.md#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4A8_REAL_002_concerns_and_verification.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_004.zh.md#1-1), lines 1 to 100

Read memory

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/cutlass_qmma_kernel_history.md)

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_00_Jam0kJF3rjMzWDgVA8Im6779__vscode-1780183079377/content.txt#1-1), lines 1 to 200

Excellent — this is the complete picture. Let me now read a few more critical documents to fill in the gaps:

Searched for files matching `**/docs/soar_2026_changes/**/*W4A8*`, 20 matches

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_W4A8_all_paths_20260521.zh.md#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_QMMA_TCGEN05.en.md#1-1), lines 1 to 50

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_QMMA_TCGEN05.en.md#50-50), lines 50 to 120

Now let me also check the v25 change doc to understand the already-shipped path:

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_REAL_FP8_GEMM.en.md#1-1), lines 1 to 80

Now let me also check the spike results document:

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md#1-1), lines 1 to 100

Now I have the complete picture. Let me compile everything into a comprehensive answer.

Completed: *Review history of CUTLASS QMMA path* (1/3)

Here's the complete analysis.

---

## 1. The Full History — What We've Done and Why

### Timeline of W4A8 Optimization

```
Apr 27 — PROPOSAL_W4A8_REAL_001: Design doc for true W4A8 (INT4 storage + FP8 MMA)
         ANALYSIS spike: CUTLASS W4-FP8 microbench → 156 TFLOPS (vs 250 TF target) → RED
         CHANGE_W4A8_001 (mislabeled W8A8): INT4→FP8 at load time → +118% regression

May 15 — PROPOSAL_W4A8_REAL_002: Verify old path, assess FP8 accuracy risk
May 16 — Build true W4A8 using TWO-STEP approach (dequant + cutlass FP8 GEMM)
May 18 — v25 SHIPPED: 81.07% accuracy, −9% speed, C=1.0 ✅

May 18-28 — PARALLEL: Attempt fused QMMA kernel (CHANGE_W4A8_QMMA_TCGEN05)
            ❌ Blocked by: PTX syntax errors, TMA descriptor issues, M=128 tile contract

May 21 — RESEARCH_W4A8_all_paths: 6-path evaluation
         Path A (CUTLASS SM120 mixed-input builder) = best path, 2-3 weeks
         Path B (fix manual PTX kernel) = high risk
         Path C (port SM100 UMMA to SM120) = promising, 1-2 weeks
May 22 — CUTLASS QMMA kernel (w4a8_fp8_qmma_sm120.cu) written but NEVER compiled
May 28 — Small-M fallback fix → fused kernel's M=128 tile contract confirmed as decode killer
May 31 — TODAY: We're debating whether to benchmark the CUTLASS QMMA kernel
```

### Key Insight: v25 Already Ships a Working Solution

The **two-step approach** (separate INT4→FP8 dequant + cutlass FP8 blockwise GEMM) is already in the v25 tarball and achieves:
- **81.07% accuracy** (+1.78pt vs baseline, C=1.0)
- **−8-9% speed regression** across all tiers
- **No PTX/TMA complexity** — uses proven CUTLASS APIs

The v25 speed regression comes from the **dequant overhead** (extra kernel launch + SMEM round-trip), not from the GEMM itself.

---

## 2. What is w4a8_fp8_qmma_sm120.cu?

### Design

It's a **CUTLASS SM100 mixed-input collective** kernel, written by us on May 22. It uses NVIDIA's `sm100_mixed_input_umma_builder.inl` (backward-compatible with SM120) to:

1. **ElementA = INT4 weights** (narrow/tuple type with packed scales)
2. **ElementB = FP8 activations** (wide type)
3. **CUTLASS handles**: TMA data movement, warp specialization, INT4→FP8 dequant, tcgen05 QMMA
4. **Output**: BF16 via CUTLASS epilogue

The kernel is **~220 lines of C++ template code** — much cleaner than the manual 440-line PTX kernel. It follows NVIDIA's recommended path for SM120 mixed-input GEMM.

### Why It Was Never Finished

**It was never compiled.** The 10 commits on May 22 were fixing compilation errors against the CUTLASS API, but the file was never added to any CMakeLists.txt. The standalone build (CMakeLists_standalone.txt) still compiles w4a8_fp8_qmma.cu (the manual PTX kernel), not this CUTLASS version.

The team pivoted because:
1. The **two-step v25 path was already working** and shipping
2. The CUTLASS QMMA path required **CUTLASS header dependencies** in the standalone build
3. **Research showed Path A (CUTLASS mixed-input builder) needed 2-3 weeks** of work
4. The **manual PTX kernel** (w4a8_fp8_qmma.cu) was attempted first as a faster path, but hit PTX/TMA blockers

---

## 3. Blockers That Prevented Forward Progress

| Blocker | Severity | Root Cause | Current Status |
|---------|----------|------------|----------------|
| **PTX syntax errors** | HIGH | tcgen05.alloc/st syntax, elect_one.sync format undocumented | Only affects manual kernel (w4a8_fp8_qmma.cu) — **NOT the CUTLASS version** |
| **TMA descriptor format** | HIGH | 128-byte descriptor layout unvalidated on SM120 | Only affects manual kernel — CUTLASS handles TMA internally |
| **M=128 tile contract** | HIGH | Decode M=1 pads to 128 → pathological slowdown | Fixed by Marlin fallback (iteration_004) |
| **CUTLASS build integration** | MEDIUM | w4a8_fp8_qmma_sm120.cu needs CUTLASS headers in standalone build | Not attempted — CUTLASS is already a CMake dependency of sgl-kernel |
| **Server startup timeout** | HIGH | CUDA graph capture 332s → workflow timeout | Caused by broken kernel (manual PTX version) |
| **Accuracy regression (MCQ)** | HIGH | 26.67% MCQ in full test | Root cause unclear — may be FP8 MMA precision, not dequant |

**Critical distinction**: The CUTLASS QMMA kernel (w4a8_fp8_qmma_sm120.cu) does NOT suffer from blockers 1-4. Those only affect the manual PTX kernel (w4a8_fp8_qmma.cu). The CUTLASS version was simply **never benchmarked**.

---

## 4. From First Principles: Is CUTLASS QMMA the Best GEMM Optimization?

### The Case For

1. **GEMM is ~83% of GPU time** (from profiling) — optimizing it is the highest-impact thing we can do
2. **FP8 has 2× theoretical peak** over BF16 (296 vs 148 TFLOPS) — if we can utilize it
3. **CUTLASS is NVIDIA's recommended path** — it handles TMA, warp specialization, and PTX generation automatically
4. **NVIDIA provides reference implementations**: `sm100_mixed_input_umma_builder.inl` and `examples/55_hopper_int4_fp8_gemm/`
5. **sgl-kernel already uses CUTLASS** for FP8 blockwise GEMM, MoE W4A8, and other kernels — the infrastructure exists

### The Case Against

1. **v25 already ships** with 81.07% accuracy and C=1.0. The −9% speed is a known cost.
2. **The CUTLASS SM100 mixed-input builder was designed for Hopper**, not Blackwell. SM120 backward compatibility is not guaranteed.
3. **Even if QMMA hits 296 TFLOPS**, the end-to-end gain may be small:
   - At **decode (bs=1)**: the bottleneck is **weight HBM bandwidth** (1398 GB/s), not compute
   - Marlin reads 0.5 bytes/element from HBM → max decode throughput is bandwidth-limited
   - FP8 QMMA doesn't reduce HBM traffic — weights are still INT4 (0.5 bytes/element)
   - **Decode won't benefit at all**
4. **At prefill (large M)**: compute-bound, so FP8's 2× peak helps. But prefill is already fast.
5. **Accuracy risk**: FP8 e4m3 has only 3 mantissa bits. The v25 two-step path already showed MCQ degradation (26.67% in the fused kernel test). The CUTLASS QMMA path uses the same FP8 precision — the accuracy issue may be fundamental.

### The Real Question

The GEMM optimization priority is correct, but the question is: **what type of GEMM optimization gives the best ROI?**

| Optimization | Effort | Expected Gain | Risk |
|-------------|--------|---------------|------|
| **CUTLASS QMMA benchmark** | 1-2 days | Unknown (could be 0-100%) | Low |
| **Register-only LOP3 dequant** (Phase 1) | 3-5 days | 2-3× dequant speed | Medium |
| **Marlin tile tuning** | 1-2 days | 5-15% | Low |
| **v25 dequant kernel optimization** | 3-5 days | 5-10% (reduce dequant overhead) | Low |

---

## 5. Recommendation

**Benchmark the CUTLASS QMMA kernel first** (1-2 days). Since it's already written, the cost is minimal:
1. Add w4a8_fp8_qmma_sm120.cu to the standalone CMakeLists
2. Compile on fcloud
3. Run the same micro-benchmark we used for the manual kernel

**Expected outcomes**:
- **≥100 TFLOPS**: CUTLASS QMMA works — proceed to integrate into the runtime
- **<50 TFLOPS**: CUTLASS QMMA has the same SMEM bottleneck as the manual kernel — the problem is architectural, not implementation-specific
- **Compile/runtime errors**: SM100 mixed-input builder is NOT backward-compatible with SM120 — abandon this path

Either way, the result directly informs our next direction. Would you like me to proceed with benchmarking the CUTLASS QMMA kernel?
