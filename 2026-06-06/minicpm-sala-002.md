Before applying, quick sanity check on the `is_none()` branches in extend output processing to confirm there's nothing MedusaWorker actually needs from that path:  

Searched for regex `def process_batch_result_prefill|def process_batch_result_decode|spec_algorithm.is_none` (`**/python/sglang/srt/managers/scheduler_output_processor_mixin.py`), 20 results

Searched for regex `spec_algorithm|spec_info` (`**/python/sglang/srt/managers/scheduler_output_processor_mixin.py`), 16 results

All `spec_algorithm.is_none()` branches are inside `process_batch_result_decode`, not `_prefill` — so leaving `spec_algorithm=NGRAM` during EXTEND won't affect prefill output processing. Apply the fix:  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#238-238), lines 238 to 252

Searched for text `batch.spec_algorithm = SpeculativeAlgorithm.NONE` (`**/python/sglang/srt/speculative/medusa_worker.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#218-218), lines 218 to 237

Replacing 7 lines with 12 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add -u && git commit -m "medusa iter 4: drop spec_algorithm=NONE reset in EXTEND (fix CHANGE_0160/0161 bonus-slot +1)" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1 | tail -5 && for i in 1 2 3 4 5 6 7 8 9 10; do python3 scripts/fcloud/fcloud_exec.py exec 'echo ready' 2>&1 | tail -3 | grep -q ready && echo "READY after ${i}0s" && break; sleep 10; done

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 900 'cd /root/submission_sim && bash ./preflight_drive.sh ngram 2>&1 | tail -8' 2>&1 | tail -12

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 900 'cd /root/submission_sim && bash ./preflight_drive.sh medusa 2>&1 | tail -8' 2>&1 | tail -12

Medusa dump now `52938B` (vs ngram `52937B`) — same size, no KV leak crash. Strong signal. Diff:   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 120 'cd /root/sglang-minicpm && git pull --ff-only origin mixed_minicpm_cudagraph 2>&1 | tail -2 && python3 benchmark/soar/demo_sala/preflight_diff.py --ngram /tmp/dump_ngram.pkl --medusa /tmp/dump_medusa.pkl 2>&1 | tee /tmp/iter4_diff.txt | tail -80' 2>&1 | tail -90

**ZERO DIFFERING FIELDS across all 4 phases.** 🎯

Let me also verify pre_prepare_for_verify is fully equal:   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 60 'grep -E "PHASE|status|EQUAL|DIFFER|TOTAL" /tmp/iter4_diff.txt' 2>&1 | tail -60

**Every single field in every phase EQUAL.** MedusaWorker is now byte-for-byte equivalent to NgramWorker on the K=1 fallback path (Stage 3a, head_pred=0).

Pause fcloud, document §5.5 + close-out, then we'll need user confirmation for the final accuracy/speed eval.    

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -3

Paused. Now write §5.5 (iter 4 result) for EN+ZH:   

Searched for text `Followup once preflight diff hits zero` (`**/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md`), 1 result

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md#275-275), lines 275 to 285

Replacing 7 lines with 55 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md)


```

```
Now the ZH mirror:   

Searched for text `运行完整 accuracy 测试` (`**/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md`), 1 result

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md#275-275), lines 275 to 285

Replacing 5 lines with 53 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.{en,zh}.md && git commit -m "preflight: CHANGE_0165 §5.5 iter4 result (0 diff — preflight loop converged)" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Completed: *Iter4: drop spec_algorithm reset in MedusaWorker EXTEND → 0 diff* (1/2)

## Iter 4 — **PRE-FLIGHT LOOP CONVERGED** 🎯

**Diff progression**:

| iter | code change | diff fields |
|---|---|---|
| 1 | broken baseline (ndt=1) | 13 |
| 2 | ndt=2 + kernel retrive_* + full mask | 5 |
| 3 | + pre_prepare_for_verify phase | 5 (+3 at new phase) |
| **4** | **drop `spec_algorithm = NONE` reset in EXTEND** | **0** ✅ |

All 4 phases × 25 fields EQUAL between NgramWorker and MedusaWorker on the K=1 Stage 3a fallback. Bonus: the post-serve KV leak (2 slots, observed in iter 2 & 3) also vanished — same root cause. Long-standing CHANGE_0160 / CHANGE_0161 "+1" bug is resolved.

**Commits this round**: `0e4634c1d` (one-line code fix) + `757ea5a6d` (§5.5 EN+ZH docs).

**fcloud paused.**

---

**Next** (BLOCKED on your explicit confirm per project rules): final accuracy + speed eval. Plan:

1. Resume fcloud → `fcloud_workflow.py full` (sync + restart-server WITH cuda-graph + torch-compile re-enabled via `SGLANG_SERVER_ARGS`, i.e. NOT `preflight_drive.sh`) + accuracy eval.
2. `fcloud_workflow.py speed --variant all` — S1 / S8 / Smax.
3. Compare to Test 12 baseline (S1=121.71s, accuracy 79.29%, normalized 99.11%, C=1.0). Expect medusa K=1 ≥ ngram speed; trained heads (Stage 4) would improve further.
4. Pause fcloud + update `TEST_RESULTS_TRACKING.md`.

Note on env launch: the `SOAR_PREFLIGHT_DUMP_PATH` env var must be **unset** for the eval run (currently disabled by default in `prepare_env.sh`, so this is automatic — `preflight_drive.sh` is the only thing that sets it). Awaiting your "go" for the final eval.  



