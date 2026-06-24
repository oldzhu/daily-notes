I have the approved slice and a concrete hypothesis: the A fragment register order is wrong for SM120 `m16n8k32.row.col`, so I’m changing that one block first and then validating with the existing focused probes.

This batch grabs the one relevant memory note and the nearby kernel lines so the edit stays narrow and grounded in the row-fold evidence.

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#82-82), lines 82 to 126

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/dead_ends.md)

I’m checking whether this repo already captured the exact SM120 A-fragment lane order; if not, I’ll use the smallest reversible remap implied by the row-fold signature.

Searched for regex `a\[0\]=A\[t/4\]|row1 = row0 \+ 8|m16n8k32 FP8 row-major` (`**/*.{cu,cuh,md,txt}`), 5 results

Searched for regex `m16n8k32.*A fragment|PTX ISA layout|a_regs\[0\]|a_regs\[1\]` (`**/docs/**`), no results

I didn’t find a better local reference for the lane contract, so I’m taking the smallest testable fix: reorder the A registers to interleave upper and lower row halves by K-half, which is the change most consistent with the observed row-fold bug.

Generating patch (23 lines) in [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu)


```

```
The first edit is in. I’m validating the touched slice immediately with a syntax/type check and then I’ll rebuild just the kernel library path needed for the row-fold probe.

Checked [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu), no problems found

Searched for regex `libw4a8_fused_gemm.so|w4a8_fp8_fused_gemm|load_library\("/root/submission_sim/libw4a8_fused_gemm.so"\)|nvcc.*w4a8` (`**/*.{py,sh,md,cu,cpp,txt}`), 70 results

I found the standalone build surface for this exact library. Next I’m using that path to rebuild the touched kernel and then re-running the row-fold probe as the cheapest discriminating check.

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_QMMA_TCGEN05.en.md#100-100), lines 100 to 155

The local file check passed. I’m moving to the cheapest real validation now: sync this single kernel file to fcloud, rebuild only the standalone `libw4a8_fused_gemm.so`, and rerun the row-fold probe before anything broader.

Created 6 todos

This batch starts the instance, uploads just the touched kernel file, and confirms the remote build workspace exists so the next step is a narrow rebuild instead of a broad sync.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY'
from scripts.fcloud import fcloud_exec
url, token = fcloud_exec.load_config()
ok = fcloud_exec.upload_file(url, token, '/home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu', '/root/sglang-minicpm/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu', chunk_size_mb=1)
print('upload_ok', ok)
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls -d /root/standalone_fused /root/standalone_fused/build 2>/dev/null || true'

The remote source is updated and the standalone build tree exists. I’m rebuilding that library now and then immediately rerunning the row-fold probe against the new `.so`.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused/build && cmake --build . -- -j2 && cp libw4a8_fused_gemm.so /root/submission_sim/'

The rebuild succeeded cleanly. Next is the discriminating check: rerun the row-fold probe and see whether row 8 still collapses into row 0.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/submission_sim && source ./prepare_env.sh >/tmp/prepare_env.log 2>&1 && python3 /root/w4a8_rowfold_probe_20260526.py'

Starting: *Run row-fold probe* (3/6)

The row-fold bug is fixed in the rebuilt library. I’m now running the next two focused checks: the distinct basis probe and then the original isolate reproducer.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/submission_sim && source ./prepare_env.sh >/tmp/prepare_env.log 2>&1 && python3 /root/w4a8_basis_probe_distinct_20260526.py'

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/submission_sim && source ./prepare_env.sh >/tmp/prepare_env.log 2>&1 && cd /root/sglang-minicpm && python3 benchmark/soar/demo_sala/test_isolate.py'

The kernel fix is validated: the row-fold probe, the distinct basis probe, and the full isolate test all pass on fcloud. I’m documenting this iteration now so the repo’s change-tracking requirements stay complete.

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_002.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_002.zh.md)

I’m recording this as a focused correctness iteration, with the exact probe results that proved the row-fold bug and the one-line kernel change that fixed it.

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_003.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_003.zh.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-qmma-rowfold-fix_20260526_0950.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-qmma-rowfold-fix_20260526_0950.zh.md)

I caught one documentation link error in the chat logs. I’m fixing that now and then I’ll record the reusable kernel lesson in repo memory.

Generating patch (10 lines) in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-qmma-rowfold-fix_20260526_0950.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-qmma-rowfold-fix_20260526_0950.zh.md)


```

```

```

```
This last batch checks the repo memory layout and records the new fragment-packing lesson so we do not rediscover the same row-fold failure later.

Read memory

Inserted into memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/dead_ends.md)

I’ve finished the fix and documentation; the only remaining action is to pause the fcloud instance again so we don’t leave the shared resource running.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance

Starting: *Pause fcloud instance* (5/6)

The first pause call hit the transient gateway timeout again. I’m retrying once now.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance

Updated todo list

Everything is done and the instance is paused. I’m closing out the task state, then I’ll summarize the fix and the validation results.

Completed: *Pause fcloud instance* (5/6)

Applied the kernel fix in w4a8_fp8_qmma.cu by reordering the A-fragment registers from grouped-by-row to interleaved-by-K-half. That removed the row-fold bug where rows `8..15` of an `m16` tile were collapsing into rows `0..7`.

Validation on fcloud passed after rebuilding the standalone library and copying it into `/root/submission_sim/`. The focused row-fold probe now passes, the distinct one-hot and two-hot basis probe passes, and test_isolate.py now passes all four cases with `maxdiff=0.00`, including the previously failing random-activation cases. I also documented the iteration in CHANGE_W4A8_001_iteration_003.en.md, CHANGE_W4A8_001_iteration_003.zh.md, and the matching chat logs under CHAT_w4a8-qmma-rowfold-fix_20260526_0950.en.md and CHAT_w4a8-qmma-rowfold-fix_20260526_0950.zh.md. The fcloud instance is paused.

Natural next steps:
1. Run an end-to-end server smoke check with the rebuilt library.
2. If outputs are sane, run fcloud accuracy before doing any speed work.

