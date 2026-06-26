Starting: *Update tracking docs* (3/4)

Let me update the tracking document and the change docs, then pause the instance.

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#1-1), lines 1 to 200

That's the complete tracking doc. Now let me add the Test 1 results:






````

````
Now let me also update the optimization catalog to reflect this finding:

Starting: *Pause fcloud instance* (4/4)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance

Completed: *Pause fcloud instance* (4/4)

The fcloud instance is paused. All tasks are complete. Here's a summary of what we've accomplished and where we go next:

---

## Summary of Test 1 — Fused W4A8 REAL + Marlin Fallback

### Results
- **Accuracy**: **67.96%** (normalized 84.94%, C=0 — eliminated). MCQ collapsed to 26.67%.
- **Speed**: Fused kernel is **2-4× slower** than Marlin at every shape and M value (35-50 TFLOPS vs 80-179 TFLOPS).
- **Verdict**: The fused W4A8 REAL path is **not viable** — worse in both accuracy and speed.

### Key insight
Despite using SM120 FP8 tensor cores (296 TFLOPS peak), the fused kernel only achieves 12-17% utilization. Marlin's INT4 path on the same hardware achieves 2-4× higher throughput. The fused kernel's accuracy regression (especially MCQ) makes it unusable regardless.

### Next steps for discussion
Since the fused W4A8 path is dead, where should we focus next? Some options:

1. **Optimize the Marlin baseline further** — we already have a strong baseline (Test 12: 79.29%, S1=121.71s). What other optimizations can we apply?
2. **Speculative decoding** — the Medusa Stage 2 pass-through showed promise (80.11% acc, S1=118.28s) but Stage 3 had issues. Worth revisiting?
3. **Sparse attention improvements** — the sparse path had bugs but may offer speed gains if fixed.
4. **Long-context tuning** — Tier 1 long-context changes (chunk=65K, prefill-max-req=4) showed no local cost and potential official upside.
5. **Other quantization approaches** — NVFP4 was catastrophic; any other paths worth exploring?

What direction would you like to explore?

