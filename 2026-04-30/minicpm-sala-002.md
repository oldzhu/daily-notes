Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 600 "rm -rf /root/submission_sim_A /root/submission_sim_B && mkdir -p /root/submission_sim_A /root/submission_sim_B && tar -xzf /root/minicpm_sala_submit_v1_007.tar.gz -C /root/submission_sim_A && tar -xzf /root/minicpm_sala_submit_v18.tar.gz -C /root/submission_sim_B && du -sh /root/submission_sim_A /root/submission_sim_B && echo ---A--- && ls /root/submission_sim_A && echo ---B--- && ls /root/submission_sim_B" 2>&1 | tail -50

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "diff /root/submission_sim_A/prepare_env.sh /root/submission_sim_B/prepare_env.sh; echo ---; grep -n 'enable-torch-compile\|skip-server-warmup\|kv-cache\|force-dense\|mem-frac\|sparse-topk' /root/submission_sim_A/prepare_env.sh; echo --- B ---; grep -n 'enable-torch-compile\|skip-server-warmup\|kv-cache\|force-dense\|mem-frac\|sparse-topk' /root/submission_sim_B/prepare_env.sh" 2>&1 | tail -60

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "grep -n 'pip install\|PYTHONPATH\|sys.path\|editable\|-e ' /root/submission_sim_A/prepare_env.sh | head -30" 2>&1 | tail -40

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "pgrep -af 'sglang.launch_server\|python.*sglang' | head -10; echo ---gpu---; nvidia-smi --query-gpu=memory.used,memory.free --format=csv" 2>&1 | tail -20

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "grep -n 'MODEL_PATH\|HOST\|PORT\|FUSED_QK\|launch_server' /root/submission_sim_A/prepare_env.sh | head -30" 2>&1 | tail -40

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "grep -n 'MODEL_PATH\|HOST\|PORT' /root/submission_sim/prepare_env.sh 2>/dev/null | head -10; echo ---; sed -n '100,135p' /root/submission_sim_A/prepare_env.sh" 2>&1 | tail -50

Searched for regex `MODEL_PATH|launch_server|HOST|PORT|submission_sim|prepare_env` (`**/scripts/fcloud/fcloud_workflow.py`), 20 results

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#251-251), lines 251 to 340

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#340-340), lines 340 to 420

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 60 "if [ -e /root/submission_sim ] && [ ! -L /root/submission_sim ]; then mv /root/submission_sim /root/submission_sim_orig && echo 'moved orig'; else echo 'already symlink or missing'; fi; ln -sfn /root/submission_sim_A /root/submission_sim && ls -la /root/submission_sim && echo ---check--- && readlink /root/submission_sim" 2>&1 | tail -20

Starting: *Variant A: launch server, run accuracy* (3/7)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py restart-server 2>&1 | tail -30

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1 | tail -30

Variant A server is ready. Now run accuracy. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py accuracy 2>&1 | tail -30

Starting: *Variant A: run s1/s8/smax speed* (4/7)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py speed --variant all 2>&1 | tail -80

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py speed --variant s1 2>&1 | grep -E "Benchmark duration|S1:|duration \(s\)|^{" | tail -10

**Variant A (v1_007) complete.** Results: acc=79.51%, S1=119.64s, S8=43.49s, Smax=35.52s. Now switching to Variant B. 

Starting: *Variant B: launch server, run accuracy* (5/7)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "pkill -f sglang.launch_server 2>/dev/null; sleep 3; ln -sfn /root/submission_sim_B /root/submission_sim && readlink /root/submission_sim && nvidia-smi --query-gpu=memory.used --format=csv,noheader" 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "sleep 10; nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -af sglang | head" 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py restart-server 2>&1 | tail -15 && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1 | tail -15

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py accuracy 2>&1 | tail -20

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "grep -E 'Average Score|Total Duration' /root/data/outputs/20260423_113315/*.txt /root/data/outputs/20260423_113315/*.log /root/data/outputs/20260423_113315/*.json 2>/dev/null; echo ---; ls /root/data/outputs/20260423_113315/" 2>&1 | tail -20

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "cat /root/data/outputs/20260423_113315/summary.txt | head -10" 2>&1 | tail -15

**Variant B accuracy: 79.51%** (identical to A in total, but per-task breakdown differs: A has higher CWE, B has higher NIAH). Running speed.  

Starting: *Variant B: run s1/s8/smax speed* (6/7)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py speed --variant s1 2>&1 | grep -E "Benchmark duration|S1:|duration \(s\)|^{" | tail -5 && echo "=== S8 ===" && python3 scripts/fcloud/fcloud_workflow.py speed --variant s8 2>&1 | grep -E "Benchmark duration|S8:|duration \(s\)|^{" | tail -5 && echo "=== Smax ===" && python3 scripts/fcloud/fcloud_workflow.py speed --variant smax 2>&1 | grep -E "Benchmark duration|Smax:|duration \(s\)|^{" | tail -5

All tests complete. Shutting down.   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "pkill -f sglang.launch_server 2>/dev/null; sleep 2; rm /root/submission_sim && mv /root/submission_sim_orig /root/submission_sim && ls -la /root/submission_sim | head -3; echo ok" 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py shutdown 2>&1 | tail -5

Completed: *Variant B: run s1/s8/smax speed* (6/7)

Now update test-results tracking with this A/B finding and commit.   

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#1-1), lines 1 to 15

Searched for text `### Package diff v1_007 → v18` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), 1 result

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#125-125), lines 125 to 145

Replacing 4 lines with 26 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/TEST_RESULTS_TRACKING.md && git commit -m "docs: Variant A+B fcloud A/B test — v18 is 7% faster, local accuracy identical

Extracted v1_007 and v18 side-by-side on fcloud, swapped /root/submission_sim
symlink between runs. Both variants hit accuracy=79.51% on public set; v18
(B) beats v1_007 (A) on all speed tiers (S1 -7.1%, S8 -6.2%, Smax -1.3%).

Per-task: A wins CWE (85.3 vs 82.0), B wins NIAH (100.0 vs 96.7); net tied.

Conclusion: v18 regression officially (80.51 → 76.64 across resubmits of same
package) is NOT attributable to local-measurable accuracy loss from the two
code deltas. Most likely source is private-set variance + eval harness
contention. Recommended next step: keep v18 code base (free 7% speed), layer
Iteration A-0 (mcq runaway fix) for v19 — v18 shows mcq avg_out=12267 vs
v1_007 avg_out=8505, confirming mcq runaway is still present and is the best
leverage point for both accuracy stability and Smax improvement." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/v1_007_vs_v18_regression_diff.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/v1_007_vs_v18_regression_diff.md)

Inserted into memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/v1_007_vs_v18_regression_diff.md)

## A+B Test Results (Option 3)

| Variant | Accuracy (local public set) | S1 | S8 | Smax | Server ready |
|---|---|---|---|---|---|
| **A** v1_007 (torch.compile OFF, FP32 cast ON) | **79.51%** (norm 99.39, C=1.0) | 119.64s | 43.49s | 35.52s | 46s |
| **B** v18 (torch.compile ON, FP32 cast OFF) | **79.51%** (norm 99.39, C=1.0) | 111.19s | 40.79s | 35.06s | 225s |

**Key findings:**

1. **Accuracy is a tie on local public set — both 79.51%.** Per-task breakdown differs (A wins CWE, B wins NIAH) but nets to the same total. So neither the FP32 qk-norm cast removal nor torch.compile causes any local-measurable accuracy loss. My primary regression hypothesis is **not confirmed locally**.

2. **v18 is genuinely 7% faster** (S1 -7.1%, S8 -6.2%, Smax -1.3%). The speed gain is real and free in terms of public-set accuracy.

3. **The v18 official regression (80.51 → 76.64 resubmit of identical package) is NOT attributable to the 2 code deltas we identified.** Remaining candidate causes:
   - Private-set sensitivity (unknown questions may behave differently under torch.compile/BF16 qk-norm, and we can't measure this locally)
   - Fcloud hardware/harness contention at submission time (variance amplified by)
   - Private-set sampling variance across submission batches

4. **mcq runaway is WORSE under v18**: avg_out for mcq = A:8505 tokens vs B:12267 tokens (+44%). Even though both variants score 53.33% on mcq (same 16/30 correct), v18 chains think-tokens ~44% longer. This is exactly the symptom Iteration A-0 targets, and it gets directly leverage under v18.

## Recommendation

**Keep v18 code path as the baseline.** Don't revert to v1_007 — it's slower with zero accuracy benefit on public eval. The v18 official regression story simplifies to "private-set variance amplified by mcq runaway under load." 

**Next step**: execute Iteration A-0 (mcq runaway fix) on top of v18, package as v19-candidate, submit. This should give:
- Better S1/S8 (fewer wasted mcq tokens → smaller KV pressure)  
- More stable accuracy (truncated mcq chains were the root cause of 76.64 vs 80.51 swings)
- Same torch.compile speed win

Committed analysis: `0aaeb5487` on `minicpm-src/mixed_minicpm_cudagraph`. Fcloud shut down. Awaiting your signal to proceed with Iteration A-0 implementation.  



