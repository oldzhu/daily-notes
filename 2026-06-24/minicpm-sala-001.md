Compacted conversationI’m resuming from the current kernel review state and tightening the evidence around the activation-sensitive failure before proposing the next discriminating check.Optimized tool selection



I’m checking existing memory and the exact local code slices so the next recommendation stays grounded in the current workspace state.

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/test_isolate.py)

Read memory

The current tree still matches the reviewed baseline; I’m checking repository notes for prior dead ends so the next step doesn’t repeat known failures.

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu#1-1), lines 1 to 220

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/dead_ends.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/w4a8_fp8_blockwise_abandoned.md)

I found the current kernel is still at the restored baseline. Next I’m pulling the short repo notes that matter for this exact kernel so I can separate confirmed dead ends from still-useful hypotheses.

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/dead_ends.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/w4a8_fp8_blockwise_abandoned.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/sm120_mixed_input_gemm_research.md)

The evidence still points at the operand mapping path, not dequant. I’m formalizing that into a short review outcome and a minimal next experiment so we can choose the next move cleanly.

Created 3 todos

I have enough context for a concrete review; I’m pulling exact line numbers from the current files so the conclusions map to the code precisely.

Ran terminal command:  cd /home/oldzhu/sglang && nl -ba sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu | sed -n '1,170p'

Ran terminal command:  cd /home/oldzhu/sglang && nl -ba benchmark/soar/demo_sala/test_isolate.py | sed -n '1,120p'

Ran terminal command:  cd /home/oldzhu/sglang && nl -ba docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu | sed -n '70,100p'

I’m recording one reusable debugging lesson from this kernel work so we don’t lose it when we switch back to implementation.

Inserted into memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/dead_ends.md)

The review is converged enough to make a concrete recommendation without touching code again yet. I’m closing the loop on the task list and then I’ll summarize the current diagnosis plus the smallest next experiment worth approving.

Completed: *Form next discriminating check* (3/3)

The current review points to one narrow conclusion: the remaining bug is still in the MMA operand path, not in the INT4 dequant path.

In the current kernel, dequant writes weights into shared memory as `W_fp8[n * kTileK + k]` at w4a8_fp8_qmma.cu, activations are staged separately at w4a8_fp8_qmma.cu, and the only activation-dependent logic after that is the fragment packing plus the MMA call at w4a8_fp8_qmma.cu. The isolate reproducer still says:
- ones activations pass at test_isolate.py
- random activations fail at test_isolate.py
- random weights with ones activations also pass at test_isolate.py

That pattern is hard to reconcile with “bad dequant only.” If weight decode were the primary issue, case 3 would not cleanly pass. What it does fit is a fragment-lane or K-order mismatch that uniform A hides. The two live suspects are:
1. The A/B fragment mapping around w4a8_fp8_qmma.cu and w4a8_fp8_qmma.cu.
2. The inline-PTX operand contract at w4a8_fp8_qmma.cu, because the current kernel passes accumulators both as `+f` outputs and again as explicit `f` inputs, while the local reference helper uses the simpler in-place form only at bench_w4fp8_sm120.cu.

My recommendation is to take one discriminating step, not another broad kernel rewrite: add a tiny 1-warp microtest with non-uniform, K-tagged A and B so we can prove whether the current fragment packing matches PTX’s `row.col` contract. That is the cheapest test that can actually falsify the top hypothesis. If you want, I can prepare that proposal next and then, after your approval, implement only that microtest.
