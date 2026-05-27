Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py console-token-info

Token fresh. Starting instance. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance

Ran terminal command:  cd /home/oldzhu/sglang && for i in 1 2 3 4 5 6 7 8 9 10; do echo "=== attempt $i ==="; python3 scripts/fcloud/fcloud_exec.py exec 'echo READY && hostname' 2>&1 | tail -3 && break || sleep 20; done

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py sync

Ran terminal command:  tail -30 /home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01BPeTitS6N3WGPLGeepkLaw__vscode-1778451758153/content.txt

sgl-kernel build failure is unrelated — Stage 2 changed only Python files. The existing wheel from earlier is still installed. Proceeding to server restart.  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls /root/submission_sim/sgl_kernel-*.whl 2>/dev/null || echo "no local wheel"; pip show sgl_kernel 2>/dev/null | head -3'

Good — sgl_kernel is still installed. Now restart with `SOAR_SPEC_MEDUSA=1`.  

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#1-1), lines 1 to 80

Searched for regex `restart.server|start_server|prepare_env` (`**/scripts/fcloud/fcloud_workflow.py`), 18 results

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py restart-server --env SOAR_SPEC_MEDUSA=1 --env SOAR_SPEC_MEDUSA_HEADS=1 2>&1 | tail -40

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1 | tail -20

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 200 2>&1 | tail -120

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'tail -200 /tmp/sglang_server.log 2>/dev/null || ls /tmp/*.log 2>/dev/null; echo ---; ps aux | grep -i "launch_server\|python3" | grep -v grep | head -20'

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls -la /tmp/*.log; echo ---; cat /tmp/pe.log 2>/dev/null | tail -100'

Searched for regex `server_cmd|launch_server|tee|\.log` (`**/scripts/fcloud/fcloud_workflow.py`), 4 results

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#290-290), lines 290 to 325

Searched for regex `server.logs|step_server_logs|def.*server.*log` (`**/scripts/fcloud/fcloud_workflow.py`), 9 results

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#440-440), lines 440 to 465

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import sys; sys.path.insert(0, 'scripts/fcloud')
import fcloud_exec, os, json
cfg = json.load(open(os.path.expanduser('~/.fcloud_config')))
url, tok = cfg['FCLOUD_URL'], cfg['FCLOUD_TOKEN']
for t in fcloud_exec.list_terminals(url, tok):
    print('==TERM', t['name'])
    print(fcloud_exec.tail_terminal(url, tok, t['name'], lines=300, wait=2)[-8000:])
    print()
" 2>&1 | tail -200

Ran terminal command:  cat ~/.fcloud_config | head -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import sys; sys.path.insert(0, 'scripts/fcloud')
import fcloud_exec
url='http://223.167.85.181:12369'; tok='73d015040882d31b6074aa3ba39d0c9f'
for t in fcloud_exec.list_terminals(url, tok):
    print('==TERM', t['name'])
    out = fcloud_exec.tail_terminal(url, tok, t['name'], lines=400, wait=2)
    print(out[-12000:])
    print()
" 2>&1 | tail -250

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Lz4pq6TfPoW7qDVway7Qmn__vscode-1778451758169/content.txt)

Server is alive and progressing CUDA-graph capture (6/16 batch sizes done). Decode-mode capture matches Stage 2 expectation. Just need longer wait.  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1 | tail -8

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import sys; sys.path.insert(0, 'scripts/fcloud')
import fcloud_exec
url='http://223.167.85.181:12369'; tok='73d015040882d31b6074aa3ba39d0c9f'
out = fcloud_exec.tail_terminal(url, tok, '2', lines=100, wait=2)
print(out[-6000:])
" 2>&1 | tail -80

11/16 captured at 11:11. ~5 more minutes for remaining 5 batch sizes. Let me wait again. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import sys; sys.path.insert(0, 'scripts/fcloud')
import fcloud_exec
url='http://223.167.85.181:12369'; tok='73d015040882d31b6074aa3ba39d0c9f'
out = fcloud_exec.tail_terminal(url, tok, '2', lines=80, wait=2)
print(out[-5000:])
" 2>&1 | tail -60

**Critical finding!** Stage 2 init **succeeded** (`MedusaWorker Stage 2 ready: K=1...` confirmed in log) but the first prefill request crashed in cuda_graph_runner: `output with shape [1] doesn't match the broadcast shape [7]`.

Root cause: a prefill batch (#new-token: 7) is being dispatched through `graph_runner.replay()` as if it were a decode batch. The graph was captured for `num_tokens_per_bs=1` decode shape. Let me investigate.  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner.py#2200-2200), lines 2200 to 2270

Searched for regex `is_cuda_graph|def is_decode|def is_extend|TARGET_VERIFY` (`**/python/sglang/srt/model_executor/forward_batch_info.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/forward_batch_info.py#95-95), lines 95 to 175

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/input_buffers.py#100-100), lines 100 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#770-770), lines 770 to 810

Searched for regex `spec_algorithm.is|forward_mode.*=|prepare_for_decode|spec_info` (`**/python/sglang/srt/managers/scheduler.py`), 40 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/scheduler.py#2270-2270), lines 2270 to 2310

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/scheduler.py#2180-2180), lines 2180 to 2240

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/tp_worker.py#420-420), lines 420 to 470

Searched for regex `def init_new|forward_mode\s*=` (`**/python/sglang/srt/model_executor/forward_batch_info.py`), 10 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/forward_batch_info.py#895-895), lines 895 to 935

Found the root cause. With `--speculative-algorithm MEDUSA` set and `--speculative-num-draft-tokens=2`, the cuda_graph_runner captured graphs (at num_tokens_per_bs=1 since MEDUSA isn't in the L281 spec branch) but the scheduler's spec-v1 code path passes the batch through paths that expect graph compatibility. The first prefill (7 tokens) hits `graph_runner.replay` which expects shape [1].

Per CHANGE_0154 §2 we already decided **eager-only for R1b**. The fix is to add `--disable-cuda-graph` when `SOAR_SPEC_MEDUSA=1`. This is a 1-line change to prepare_env.sh.

Let me also check torch-compile interaction.   Compacted conversation

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#390-390), lines 390 to 425

Searched for text `torch-compile` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 11 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#270-270), lines 270 to 300

Replacing 4 lines with 19 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)


```

```
Now let me append a brief addendum to CHANGE_0155 documenting the Stage 2 fcloud bring-up bug + fix. 

Ran terminal command:  wc -l /home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.en.md /home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.zh.md

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.en.md#160-160), lines 160 to 173

Doc is short, I'll append in-place.  

Replacing 6 lines with 140 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.zh.md)


```

```

```

```
Now commit + push, then ask user for approval to re-run fcloud (instance is still up — billing).   

Ran terminal command:  cd /home/oldzhu/sglang && git status --short && git --no-pager diff --stat HEAD

Ran terminal command:  cd /home/oldzhu/sglang && git add benchmark/soar/demo_sala/prepare_env.sh docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.en.md docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.zh.md && git commit -m "fix(medusa-r1b): force eager mode when SOAR_SPEC_MEDUSA=1 (Stage 2 bring-up)

First fcloud restart with Stage 2 MedusaWorker confirmed worker init
(MedusaHeads ~32 MiB BF16 on cuda:0) but the first prefill request
crashed in graph_runner.replay:

  RuntimeError: output with shape [1] doesn't match the broadcast shape [7]

Root cause: cuda_graph_runner.py L281 only sets num_tokens_per_bs=
speculative_num_draft_tokens for EAGLE/STANDALONE/NGRAM. MEDUSA falls
through to default (=1), so 16 graphs were captured at decode shape.
But spec_algorithm != NONE makes scheduler.run_batch take the spec-v1
branch and pass ScheduleBatch directly into model_worker; combined with
--enable-torch-compile, prefill flows into graph_runner.replay where
raw_num_token=1 cannot accept the 7-token prefill input_ids.

Per CHANGE_0154 \u00a72 we already committed to eager-only in R1b (CUDA
graph capture for verify path is deferred to R1c). Implement that here:
when SOAR_SPEC_MEDUSA=1, strip --enable-torch-compile and
--torch-compile-max-bs N from SGLANG_SERVER_ARGS, and append
--disable-cuda-graph defensively.

Expected impact: Stage 2 will be slower than v22 baseline (no graph,
no compile), but accuracy should be byte-identical. Speed regression
is removed in R1c.

Doc updates: appended \u00a710 'Stage 2 fcloud bring-up addendum' to both
EN and ZH CHANGE_0155." && git --no-pager log -1 --stat

Ran terminal command:  cd /home/oldzhu/sglang && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -10

Pushed (`fc2fc8ab1`). The fcloud instance is still running and burning billing — let me ask whether to re-test now or pause first. 

Asked a question (next-action)

User not available. Per the cost-saving rule and the explicit "Always ask the user for explicit approval before starting any fcloud automated test", I cannot start a new test round autonomously. The previous round is complete (we have the crash trace we needed for analysis). Pausing the instance now to stop billing.    

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -20

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -10

Instance paused (任务已暂停). 

## Summary

**Stage 2 bring-up bug diagnosed, fixed, committed, pushed.**

| Item | Status |
|------|--------|
| Diagnosis | Done — root cause: `cuda_graph_runner` captured at `num_tokens_per_bs=1` (MEDUSA missing from spec branch) but scheduler+`--enable-torch-compile` routed prefill into `graph_runner.replay` → 7-token prefill couldn't broadcast into shape-[1] buffer |
| Fix | prepare_env.sh — when `SOAR_SPEC_MEDUSA=1`, strip `--enable-torch-compile` / `--torch-compile-max-bs N`, append `--disable-cuda-graph` (matches CHANGE_0154 §2's eager-only commitment for R1b) |
| Doc | Appended §10 "Stage 2 fcloud bring-up addendum" to CHANGE_0155 EN and CHANGE_0155 ZH |
| Commit | `fc2fc8ab1` pushed to `minicpm-src/mixed_minicpm_cudagraph` |
| fcloud | Paused (任务已暂停) — no billing |

**Expected on next re-test (requires user approval):**
- Accuracy: byte-identical to v22 baseline (Stage 2 is pure pass-through)
- Speed: **noticeably slower** than v22 (eager mode, no torch-compile, no CUDA graph). Expected 1.3–2× regression on S1/S8/Smax. This is acceptable for Stage 2 validation; CUDA-graph capture for verify is restored in R1c.

**Next round (waiting for user "go"):** `fcloud_workflow.py start-instance` → `sync` → `restart-server` → `accuracy` → `speed --variant s1` → `pause-instance`. Then proceed to Stage 3 (full verify+rewind) per CHANGE_0154 §5. 



