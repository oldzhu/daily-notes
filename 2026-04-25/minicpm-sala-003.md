All speed benchmarks complete. Results:
- **S1: 111.48s** (baseline 110.58s → +0.8%)
- **S8: 40.42s** (baseline 40.53s → -0.3%)
- **Smax: 33.53s** (baseline 33.58s → -0.1%)

**CHANGE_0125 is neutral** — no speed impact. The new SM120 tiles compiled but were never selected by the scoring function. Let me record results and shutdown. 

Starting: *Record results in tracking doc* (7/9)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#46-46), lines 46 to 50

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#39-39), lines 39 to 46

Replacing 3 lines with 4 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```
Now add the speed benchmark results: 

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#74-74), lines 74 to 85

Replacing 2 lines with 3 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```
Now shutdown fcloud:  

Starting: *Analyze results and decide next steps* (8/9)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py shutdown 2>&1

Completed: *Analyze results and decide next steps* (8/9)

## CHANGE_0125 Test Results Summary

**Test 27 — SM120 Marlin Tile Instantiations**

| Metric | Test 27 (CHANGE_0125) | Baseline (Test 25B) | Delta |
|--------|----------------------|---------------------|-------|
| **Accuracy** | 77.18% | 79.00% | -2.3% (variance) |
| **S1** | 111.48s | 110.54s | +0.9% |
| **S8** | 40.42s | 40.54s | -0.3% |
| **Smax** | 33.53s | 33.58s | -0.1% |

**Key findings:**
- Server log shows only **1 SM120 message** — the selected config was an **existing** tile `(thread_m_blocks=2, thread_n=64, thread_k=128)`, NOT any of the 3 new tiles from CHANGE_0125
- The new tiles `(16,8,256)`, `(16,4,256)`, `(8,8,256)` compiled successfully but the scoring function never picks them for MiniCPM-SALA's weight shapes
- **CHANGE_0125 is NEUTRAL** — no speed gain, no accuracy impact (the 77.18% drop is qa task variance at 46.67% vs typical ~56%)

**Next steps to consider:**
1. The Marlin GEMM tile approach won't yield gains without also tuning the **scoring function** to prefer the new tiles
2. Higher-impact optimization vectors from the catalog should be prioritized (e.g., operator fusion, custom GEMM kernels for specific shapes)
3. Consider reverting CHANGE_0125 to reduce sgl-kernel binary size since the tiles aren't used 



