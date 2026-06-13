Accuracy at **79% (119/150)** — still running, server stable this time! Let me wait for completion:

Ran terminal command:  sleep 600 && cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls /root/data/outputs/20260519_041637/ 2>&1; wc -l /root/data/outputs/20260519_041637/predictions.jsonl 2>/dev/null; cat /root/data/outputs/20260519_041637/accuracy.json 2>/dev/null' 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 && sleep 3 && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1

Ran terminal command:  sleep 30 && cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'echo "alive" && ls -la /root/data/outputs/20260519_041637/ 2>&1 && echo "---" && wc -l /root/data/outputs/20260519_041637/predictions.jsonl 2>/dev/null || echo "no predictions"' 2>&1

Empty again. This confirms the pattern — W4A8 crashes under sustained concurrency=32. Let me provide final results and recommendation:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1



## W4A8 REAL — Final Verdict: ❌ NOT Submission-Viable

### Speed Results (all slower than Marlin baseline)

| Tier | W4A8 FP8 | Marlin Baseline | Δ |
|------|----------|-----------------|-----|
| S1 | 265.15s | 121.71s | **+118%** |
| S8 | 69.33s | 44.09s | **+57%** |
| Smax | 45.82s | 35.86s | **+28%** |

### Root Cause: FP8 Weight Bandwidth Penalty
FP8 weights are 8-bit vs INT4 4-bit = **2× HBM data**. The GEMM is memory-bandwidth-bound at all practical batch sizes (max-bs=24). FP8 QMMA's theoretical 296 TFLOPS advantage never materializes — requires M > 512 to become compute-bound.

### Accuracy: 3 attempts, 0 results
Server crashes at ~75-80% progress under concurrency=32. No predictions saved across 3 attempts.

### Path Forward
The **fused INT4→FP8 dequant kernel** (keep INT4 in HBM, convert to FP8 in SMEM) would fix the bandwidth issue. That's the Day 3 work that wasn't completed. For now, **revert to Marlin baseline** (`SOAR_W4A8_REAL_FP8_GEMM=0`) for v25 submission.
