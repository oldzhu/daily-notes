Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1 | tail -8

Ran terminal command:  cd /home/oldzhu/sglang && for i in $(seq 1 30); do status=$(curl -sk -o /dev/null -w "%{http_code}" "$(grep FCLOUD_URL ~/.fcloud_config | cut -d= -f2 | tr -d '"')/lab" 2>/dev/null); echo "$i $(date +%H:%M:%S) lab=$status"; if [ "$status" = "200" ] || [ "$status" = "302" ]; then break; fi; sleep 15; done

Ran terminal command:  cd /home/oldzhu/sglang && sleep 5 && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -10

Now re-quantize iter-5 (FOS=1). 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'rm -rf /root/models/MiniCPM-SALA-NVFP4-FOS && cd /root/submission_sim && source ./prepare_env.sh >/dev/null 2>&1 && SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1 SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 SOAR_GPTQ_CALIBRATION_SAMPLES=32 SOAR_GPTQ_CALIBRATION_SAMPLING=sequential SOAR_GPTQ_CALIBRATION_SEED=20260320 python3 -u preprocess_model.py --input /root/models/openbmb/MiniCPM-SALA --output /root/models/MiniCPM-SALA-NVFP4-FOS --mode nvfp4 2>&1 | tail -25'

Quant done; pct_m4=43.14% (matches iter-5). Restart server.   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py restart-server --quant-mode gptq --model-path /root/models/MiniCPM-SALA-NVFP4-FOS --env SOAR_QUANT_PROFILE=nvfp4_fos --env SOAR_NVFP4_FOUR_OVER_SIX=1 --env SOAR_TIER1_LONG_CONTEXT=1 --env SOAR_TORCH_COMPILE_MAX_BS=24 2>&1 | tail -8

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1 | tail -3 && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py accuracy --quant-mode gptq --model-path /root/models/MiniCPM-SALA-NVFP4-FOS 2>&1 | grep -E "Total Duration|count=30 correct|^\[accuracy\]|ori_acc|Average|Bucket Acc" | head -30

**Iter-5 run-2 = 70.98%** (cwe 76 / fwe 85.56 / mcq 46.67 / niah 100 / qa 46.67), duration 2513.84s. PASSES ≥70% gate. vs run-1 (71.24%): −0.26pt overall — **iter-5 is reproducible**. Now run speed bench. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py speed --variant all 2>&1 | tail -40

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py speed --variant all 2>&1 | grep -E "Benchmark duration|^\[speed|S1=|S8=|ERROR|FAILED|variant" | head -30

S1=173.83 / S8=46.05 / Smax=31.07. Pause now.  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && sleep 10 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Paused. Now write iter-5 run-2 doc + tracking row + commit.  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_006.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_006.zh.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_phase-b-fos-iter5-repro_20260508_1750.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_phase-b-fos-iter5-repro_20260508_1750.zh.md)

Replacing 1 lines with 2 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_006.en.md docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_006.zh.md docs/soar_2026_changes/chat/CHAT_phase-b-fos-iter5-repro_20260508_1750.en.md docs/soar_2026_changes/chat/CHAT_phase-b-fos-iter5-repro_20260508_1750.zh.md docs/soar_2026_changes/TEST_RESULTS_TRACKING.md && git commit -m "docs(0151): iter-7 NVFP4-FOS iter-5 reproducibility + speed bench

Iter-7 = run-2 of iter-5 recipe + S1/S8/Smax speed benchmark.

Accuracy: 70.98% (run-2) vs 71.24% (run-1), Δ -0.26pt overall.
Reproducible at the overall level; per-task variance ±10pt
(cwe +5.33, niah +10.00, fwe/mcq -6.66, qa -3.33).

Speed (NVFP4-FOS / Test 12 GPTQ): S1 173.83/121.71, S8 46.05/44.09,
Smax 31.07/35.86. NVFP4 +42.8% slower at S1, +4.4% slower at S8,
-13.4% faster at Smax. Score head-to-head (Final = S1*40% + S8*30% +
Smax*30%): GPTQ 96.0 vs NVFP4 86.7. With norm-acc 88.73% < 97% the
correctness coefficient is 0 -> Final score 0.

Decision: NVFP4-FOS NOT submission-ready. GPTQ baseline (Test 12)
remains submission. Park NVFP4-FOS work; resume GPTQ optimization
catalog (OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md).

Tokenizer-save fix from CHANGE_0151_005 (commits 39c0045c5+83921b207)
also validated on a real FOS=1 quant -- no manual cp required." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Iter-7 complete and committed (`d48793563`). Instance paused.

**Summary**:
- **Iter-5 reproducibility confirmed**: run-2 = 70.98% vs run-1 = 71.24% (Δ −0.26pt overall). Per-task variance ±10pt (failure mass redistributes but total stays stable).
- **Speed bench (first NVFP4-FOS S1/S8/Smax)**: S1=173.83 / S8=46.05 / Smax=31.07. vs Test 12 GPTQ (121.71 / 44.09 / 35.86): NVFP4 +42.8% slower at S1, +4.4% slower at S8, **−13.4% faster at Smax**.
- **Performance head-to-head** (Final = 40·S1 + 30·S8 + 30·Smax): GPTQ **96.0** vs NVFP4 **86.7** — S1's 40% weight dominates.
- **Submission verdict**: NVFP4-FOS norm-acc 88.73% < 97% → C=0 → final score 0. **NOT submission-ready.** GPTQ baseline (Test 12) remains the submission.
- **Tokenizer-save fix validated** on a real FOS=1 quant — no manual cp needed.
- **Decision**: park NVFP4-FOS, resume GPTQ catalog (OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md).  



