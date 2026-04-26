Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#1-1), lines 1 to 80

Read memory [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/sm120_hardware.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#80-80), lines 80 to 180

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/nvfp4_test_result.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/marlin_gemm_sm120_research.md)

Here's an honest analysis based on all current research:

## GEMM Improvement Options — Current State

### What's been eliminated

| Approach | Status | Reason |
|----------|--------|--------|
| Marlin SM80 tile changes | ❌ Dead end (CHANGE_0125) | SM80 MMA path, scorer correct — cannot access SM120 HW |
| NVFP4 (Test 21) | ❌ Failed | 12% accuracy — model loses coherence, infinite `<think>` loops |

### The core constraint

**Decode is memory-bound.** At M=1-8, loading W4 weights from GDDR7 is the bottleneck — 148 TFLOPS of BF16 compute is far more than needed. This means:
- Switching to W8 FP8 doubles bandwidth → decode gets **slower**, not faster
- Higher compute throughput (FP8, FP4) only helps **prefill** (M≥64, compute-bound)

### Three realistic GEMM paths forward

**Path 1: W4→FP8 dequant in-kernel (W4A8 GEMM)** ⭐ Most promising
- Keep weights stored as W4 INT (same bandwidth as now)
- Dequant to FP8 in L1/shared memory, then run FP8 tensor cores at **296 TFLOPS**
- Net: Same memory bandwidth, 2× compute for prefill
- Matches the competition's TRT-LLM hint (FP8 block-scale GEMM, L133-L171)
- Effort: High — requires new kernel or integrating TRT-LLM SM120 GEMM
- Gain: **~10-20% on prefill** (prefill = 85.3% of GEMM time)

**Path 2: CUTLASS 3.x SM120 warp-level MMA**
- Replace Marlin's SM80 `mma.sync.aligned.m16n8k16` with SM120 native warp-level MMA
- Same W4 format, but uses SM120-optimized instruction scheduling + TMA
- Effort: Very high — rewrite kernel from scratch
- Gain: Uncertain, but "similar to Hopper" suggests meaningful improvement

**Path 3: Speculative decoding (reduce GEMM count)**
- Rather than making each GEMM faster, do fewer GEMMs total
- Already researched (Eagle3, ngram). Acceptance rate determines gain.
- Key: S1 (40% weight) is decode-dominated → speculative decoding directly targets it
- Effort: Medium — Eagle3 integration was researched, ngram is simpler
- Gain: **20-50% if acceptance rate ~0.6+** (reduces decode step count)

---

### Recommendation

The clearest path is a **two-track approach**:

1. **Near-term (1-2 weeks, lower risk)**: Investigate TRT-LLM SM120 FP8 block-scale GEMM integration. The competition organizers specifically pointed here — this suggests it's achievable and intended. It targets prefill (85.3% of GEMM) without hurting decode bandwidth.

2. **Parallel (medium risk)**: Ngram speculative decoding (simpler than Eagle3, no model changes) — directly reduces decode GEMM count, helps S1 most.

Want me to look at either of these in more detail — e.g., investigate TRT-LLM's SM120 GEMM code and evaluate what would be needed to integrate it, or assess ngram speculative decoding feasibility? 


