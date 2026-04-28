Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py sync

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py restart-server --quant-mode gptq

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py accuracy

Test 30 result: **acc_ori=77.96% (normalized 98.32%, C=0.96)** — slightly worse than Test 29 (78.73%). Critically:

| Task | Test 29 (e5m2) | Test 30 (e4m3) |
|---|---|---|
| cwe | 56.67 | **82.00** ↑ |
| fwe | 83.67 | **97.78** ↑ |
| mcq | 96.67 | **53.33** ↓↓ |
| niah | 100 | 100 |
| qa | 56.67 | 56.67 |
| **Overall** | **78.73** | **77.96** |

**mcq collapsed from 96.67→53.33**, avg_out=11,368 tokens — runaway thinking chains are *worse* with e4m3, not better. But cwe+fwe improved substantially. Huge per-task variance suggests eval noise plays a role too.

Hypothesis partially refuted: KV precision direction is unclear. Let me shutdown first, record, then recommend. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py shutdown

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#1-1), lines 1 to 60

Replacing 3 lines with 4 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```
## Test 30 Summary — e4m3 FAILED

| Metric | Test 29 (e5m2) | Test 30 (e4m3) | Δ |
|---|---|---|---|
| acc_ori | 78.73% | **77.96%** | -0.77 |
| normalized | 98.42% | 98.32% | ≈same |
| C | 0.96 | 0.96 | same |
| Duration | 3005.79s | 2878.90s | **-4.2% faster** |
| mcq | 96.67 | **53.33** | **-43** ↓↓ |
| cwe | 56.67 | 82.00 | +25 ↑ |
| fwe | 83.67 | 97.78 | +14 ↑ |
| niah/qa | 100/56.67 | 100/56.67 | = |

**Key findings**:
1. **e4m3 did NOT fix mcq runaway** — mcq avg_out=11,368 (vs 11,143 in Test 28). KV dtype mantissa hypothesis refuted.
2. **Huge per-task swings** (cwe +25, mcq -43, fwe +14) with near-identical overall score → eval has massive per-task variance (~±25 points per task). Overall ±1% is the signal floor.
3. **Speed improved ~4%** (3005→2878s) with e4m3 — interesting, may indicate different numerical paths taken.
4. C=0.96 preserved (both tests sit in the 98-99% normalized band).

**Analysis**: The 2-pt local/official gap is NOT explained by KV dtype. Running Test 31 (BF16 KV) next would only confirm what e4m3 already showed — KV dtype is not the dominant factor. The real culprit is likely:
- **Eval variance + proximity to C=0.96/0.92 threshold boundary** (normalized 98% = cliff between 0.96 and 0.92)
- **v19 aggressive scheduling** under concurrency=32 eval amplifying this variance

**Recommendation — skip Test 31, go directly to Test 32** (revert scheduling to v18): `--prefill-max-requests 4→1`, `--schedule-conservativeness 0.8→1.0`, `--chunk 65536→32768`. v18 config gave acc_ori=78.71-80.51 on official, while v19 gave 76.0-76.67. This is a **direct reproduction test** of what the official tested last time.

fcloud is shut down. Awaiting your decision: 
- **(a)** proceed with Test 32 (scheduling revert) 
- **(b)** run Test 31 (BF16 KV) anyway for data 
- **(c)** wait for your v18 official resubmit result before next test  



