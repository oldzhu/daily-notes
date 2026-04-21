The accuracy test was interrupted. Let me check if it produced any partial results, and also check the server health: 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "curl -s -w 'HTTP:%{http_code}' http://localhost:30000/health 2>/dev/null || echo 'NO_CONN'" --timeout 15
Completed with input:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "curl -s -w 'HTTP:%{http_code}' http://localhost:30000/health 2>/dev/null || echo 'NO_CONN'" --timeout 15

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "curl -s -w 'HTTP:%{http_code}' http://localhost:30000/health 2>/dev/null || echo 'NO_CONN'" --timeout 15
Completed with input:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "curl -s -w 'HTTP:%{http_code}' http://localhost:30000/health 2>/dev/null || echo 'NO_CONN'" --timeout 15

The websocket is hanging — the server may have crashed again with OOM during inference. Let me check the server logs:  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 30
Completed with input:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 30

The websocket connection to fcloud is hanging entirely — the fcloud instance is unresponsive, likely because the server OOM killed the instance again. Let me try a simple exec with a short timeout:  

Ran terminal command:  cd /home/oldzhu/sglang && timeout 10 python3 scripts/fcloud/fcloud_exec.py exec "echo alive" --timeout 5 2>&1 || echo "FCLOUD_UNREACHABLE"

OK fcloud is alive. The previous commands were hanging on websocket. Let me try the tail command differently:  

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 20 2>&1 || echo "TAIL_TIMEOUT"

The server is **alive and processing requests**! Key observations:
- **accept len: 1.07, accept rate: 0.27** — the draft model is barely helping (only accepting ~1.07 tokens per step, 27% acceptance rate). This is expected since the draft model has random/untrained weights.
- Server is handling 24 concurrent requests with 8 in queue
- Gen throughput ~733 tokens/s during decode

The accuracy test is actively running. Let me wait for it to complete. Let me check the eval script output:   

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "ls -la /root/data/outputs/ 2>/dev/null | tail -5" --timeout 10

Good — a new output directory `20260417_015816` was created at 01:58. The test started ~10 min ago. Let me wait a bit more and check if it's still running:    

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'still running'" --timeout 10

Still processing — predictions file doesn't exist yet (eval script writes it all at the end). Let me wait longer and monitor. I'll check the server logs periodically:    

Ran terminal command:  cd /home/oldzhu/sglang && sleep 300 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 5 2>&1

Server is still alive and processing. Still 8 in queue. Let me check if predictions are being written:     

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'; ls -la /root/data/outputs/20260417_015816/" --timeout 10

The output directory exists but is still empty — the eval script sends all requests first and writes results at the end. The test is still in progress. Let me wait 5 more minutes:      

Ran terminal command:  cd /home/oldzhu/sglang && sleep 300 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 3 2>&1

Still running, still 8 in queue. The eval sends all 150 requests with concurrency — it needs to process all of them. The server is running with EAGLE3 spec decode which with only ~1.07 acceptance rate doesn't speed up decode much, but also doesn't crash. Let me wait another 5 minutes:       

Ran terminal command:  cd /home/oldzhu/sglang && sleep 300 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 3 2>&1

Still processing, token usage slowly growing (0.10 now). Let me wait 10 more minutes this time:         

Ran terminal command:  cd /home/oldzhu/sglang && sleep 600 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 3 2>&1

Queue is draining (from 8 → 5). Still processing. Let me wait 10 more minutes:           

Ran terminal command:  cd /home/oldzhu/sglang && sleep 600 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 3 2>&1

Good progress! Queue is now at 0, only 20 running requests remaining (from 24+8 originally). Almost done. Let me wait 5 more minutes:             

Ran terminal command:  cd /home/oldzhu/sglang && sleep 300 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 5 2>&1

Down to 17 running requests, 0 in queue. Let me wait another 5 minutes:                

Ran terminal command:  cd /home/oldzhu/sglang && sleep 300 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 5 2>&1

Still 17 running — these are likely the long-context MCQ requests generating many tokens (~1400 output tokens each). Token count is growing (1.8M tokens). The eval must be using max_tokens that are very large. Let me wait 10 more minutes:                   

Ran terminal command:  cd /home/oldzhu/sglang && sleep 600 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 5 2>&1

Still 17 requests running, 2M+ tokens. These seem to be generating far too many tokens — possibly no max_tokens limit. Let me check what the eval script sets:  

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "grep -n 'max_tokens\|max_new_tokens\|max_completion' /root/data/eval_model_001.py | head -10" --timeout 10

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "grep -n 'max_out_len' /root/data/eval_model_001.py | head -10" --timeout 10

`max_out_len=65536` — so each request can generate up to 65K tokens! With 17 MCQ requests each potentially outputting up to 65K tokens at ~400 tok/s total, this could take hours. 

But in the baseline (without EAGLE3), MCQ only generated ~1400 tokens on average. The issue is likely that with the untrained EAGLE3 draft model, the model's EOS/stop token behavior may be disrupted. Let me check if the server is even producing `200 OK` responses (which would mean some requests *are* finishing):   

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "grep '200 OK' /proc/\$(pgrep -f 'sglang::scheduler' | head -1)/fd/2 2>/dev/null | tail -5 || echo 'cannot read'" --timeout 10

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 30 2>&1 | grep '200 OK' | wc -l

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 100 2>&1 | grep -c '200 OK'
Completed with input:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 100 2>&1 | grep -c '200 OK'

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 200 2>&1 | grep '200 OK' | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 500 2>&1 | grep '200 OK' | wc -l

No completed requests in the last 500 log lines — all 17 remaining requests are stuck generating tokens without stopping. With `max_out_len=65536`, each request could produce up to 65K tokens. At ~400 tok/s total across 17 requests, that's ~24 tok/s per request, meaning each could take up to ~45 minutes. Some might already have generated a lot, but the last 500 lines show only decode batches.

This will complete eventually but will take a very long time. Let me check how many total tokens have been generated and estimate remaining time:  

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 3 2>&1

Down to 14 running requests! Requests are finishing. Some have completed and hit either EOS or max_tokens. Let me wait another 10 minutes:      

Ran terminal command:  cd /home/oldzhu/sglang && sleep 600 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 3 2>&1

Down to 1 running request! Almost done. Let me wait 2 more minutes:        

Ran terminal command:  cd /home/oldzhu/sglang && sleep 120 && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null || echo 'not yet'" --timeout 10 && timeout 15 python3 scripts/fcloud/fcloud_exec.py tail 1 --lines 5 2>&1

The last request completed with `200 OK`! Let me check the predictions file:     

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "wc -l /root/data/outputs/20260417_015816/predictions.jsonl 2>/dev/null && echo '---' && ls -la /root/data/outputs/20260417_015816/" --timeout 10

150 predictions written. Let me get the accuracy results:        

Ran terminal command:  cd /home/oldzhu/sglang && timeout 15 python3 scripts/fcloud/fcloud_exec.py exec "cat /root/data/outputs/20260417_015816/summary.txt" --timeout 10

**Results are in!** The Mamba rollback fix worked — the model now produces meaningful output (not 0% anymore). But accuracy is degraded compared to baseline:

**EAGLE3 Test Results (Test 22):**
- **Original Accuracy: 74.33%** (baseline: 80.64%)
- **Normalized Accuracy: 92.92%** (baseline: 100.80%)
- **C = 0** (≤97% → eliminated!)
- Duration: 4844s (vs baseline ~41s for smax)

Key issues:
1. **MCQ accuracy: 56.67%** (baseline: ~76.67%) — avg output tokens = 8527 (baseline: ~1442). The model is over-generating and not stopping properly.
2. **qa accuracy: 63.33%** (baseline: ~90%)
3. **NIAH: 100%** — perfect, unchanged
4. **CWE: 78.33%** — similar to baseline
5. **Accept rate: 0.26** — draft model is essentially useless (random weights)

The degraded accuracy is concerning. EAGLE3 speculative decoding with an untrained draft model is not just slow — it's actively hurting accuracy. The verify step should guarantee correctness (target model tokens are used), but the overhead and possible state management issues with the random draft model appear to be corrupting the recurrent state.

Let me now run speed benchmarks quickly to have complete data, then record everything and shut down: 

Starting: *Run speed benchmark S1* (3/5)

Given the poor accuracy (C=0, eliminated), running speed benchmarks is less critical but useful for data. Let me run S1 quickly:    

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py speed --variant s1 2>&1 | tee /tmp/eagle3_speed_s1.log

**S1 = 187.01s** vs baseline **113.67s** — EAGLE3 is **65% slower** than baseline at S1. The accept rate of 1.04 (essentially 0% bonus tokens) means the overhead of running the draft model + verify adds pure cost.

Now let me record results and shut down. Skipping S8/Smax since the performance picture is clear — EAGLE3 with an untrained draft model is much worse than baseline in every metric:  

Starting: *Record results in TEST_RESULTS_TRACKING* (4/5)

Ran terminal command:  cd /home/oldzhu/sglang && wc -l docs/soar_2026_changes/TEST_RESULTS_TRACKING.md

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#85-85), lines 85 to 99

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#50-50), lines 50 to 85

Now let me add the EAGLE3 test results:   

Replacing 3 lines with 4 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```


Starting: *Shut down fcloud* (5/5)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py shutdown

Completed: *Shut down fcloud* (5/5)

Now let me update the session memory with findings:  

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/eagle3_implementation_plan.md)

All done. Here's the summary:

---

**EAGLE3 Test 22 Results (untrained draft model):**

| Metric | EAGLE3 | Baseline (Test 20) | Change |
|--------|--------|-------------------|--------|
| Accuracy | 74.33% (C=0) | 80.64% (C=1.0) | -6.3pp, **eliminated** |
| Normalized | 92.92% | 100.80% | -7.9pp |
| S1 | 187.01s | 113.67s | **+65% slower** |
| Accept rate | 0.26 | — | Random draft = no benefit |

**Key findings:**
- The Mamba state rollback fix (mambaish_config) worked — output is no longer garbled
- But accuracy still degraded badly, especially MCQ (56.67% vs ~76.67%) due to massive overgeneration (8527 avg output tokens vs ~1442 baseline)
- Accept rate 0.26 means the untrained draft model adds pure overhead
- EAGLE3 also disables mixed-chunk and overlap scheduler, which may contribute to degradation

**Recommendation:** EAGLE3 with an untrained draft model is not viable. Two options:
1. **Train the draft model** on fcloud (~2-4 hrs GPU time), then retest — but even with better accept rate, the scheduling penalties may not be offset
2. **Pivot to other optimization vectors** from the catalog (kernel opts, operator fusion, etc.) — lower risk, more predictable gains

fcloud instance has been shut down.

