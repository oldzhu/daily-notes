The issue is stale server processes holding GPU memory. The `pkill` timed out. Let me try a different approach: 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "kill -9 167 50972 50973 50974 51305 2>/dev/null; sleep 2; nvidia-smi | head -20" --timeout 30
Completed with input:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "kill -9 167 50972 50973 50974 51305 2>/dev/null; sleep 2; nvidia-smi | head -20" --timeout 30

The fcloud JupyterLab API is returning 503 - we may have killed the Jupyter server too. The instance may need to be restarted. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "echo hello" --timeout 15
Completed with input:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "echo hello" --timeout 15

The fcloud JupyterLab API itself is returning 503 — likely because the previous `pkill -9 -f sglang` also killed the Jupyter terminal server process, or the GPU OOM caused instability. The fcloud instance needs to be restarted.

Before asking you to restart it, let me save our progress and update the session memory: 

Read memory

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/eagle3_implementation_plan.md)

Compacted conversation

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/eagle3_implementation_plan.md)

Now let me also review the accuracy test output that already ran. Looking at the terminal output more carefully:

There are TWO accuracy test results:
1. **First run (outputs/20260416_093021)**: 0% accuracy — this was BEFORE the mambaish_config fix
2. **Second run**: The accuracy test after the fix couldn't connect because the server had already crashed with OOM (terminal 10) and then we couldn't start it again because the old processes held GPU memory

The latest crash at terminal 8 shows:
- Server started OK, loaded model (75.78 GB avail → 22.68 GB after KV cache)
- CUDA graph capture was in progress (88% done at `bs=2`)  
- Then the accuracy test ran on terminal 8's server but it CRASHED during `forward_draft_extend` in minicpm_eagle3.py line 75 (`down_proj`) with OOM

Wait — looking at the timeline again:
- Terminal 8 started at 09:37 → it loaded model successfully and started CUDA graph capture
- But the accuracy test output from `20260416_093021` ran at 09:30... that's terminal 7's server (from the PREVIOUS session before the fix)
- Terminal 7's server died during inference with OOM at `minicpm_eagle3.py:75 down_proj`

So both servers died during inference with OOM — **the draft model's forward pass itself runs out of memory during inference even with 0.72 mem-fraction-static**. This is a memory problem, not just a stale process problem.

Let me check the actual crash more carefully: 

Starting: *Diagnose EAGLE3 OOM during inference* (1/6)

Read [](file:///home/oldzhu/sglang-eagle3/python/sglang/srt/models/minicpm_eagle3.py)

Read [](file:///home/oldzhu/sglang-eagle3/python/sglang/srt/models/minicpm_eagle3.py)

The OOM during inference looks like the draft model is processing too many tokens. Let me examine the eagle_worker flow:  

Read [](file:///home/oldzhu/sglang-eagle3/python/sglang/srt/speculative/eagle_worker.py)

Read [](file:///home/oldzhu/sglang-eagle3/python/sglang/srt/speculative/eagle_worker.py)

The fcloud JupyterLab API itself is returning 503 — likely because either the Jupyter process was affected by the kill or the instance is overloaded from the OOM crashes.

**Summary of situation:**

1. **Server terminal 7** (old code, before Mamba fix): Started OK, ran accuracy test (0% accuracy due to missing Mamba rollback), then **crashed with OOM during inference** — the draft model's `down_proj` tried to allocate 256 MiB with only 63 MiB free (82.96 GB of 83 GB in use)

2. **Server terminal 10** (new code, with Mamba fix): Failed to start because terminal 7's process (PID 50973) was still holding 76.22 GB GPU memory

3. **Server terminal 8** (second restart attempt after reducing mem-fraction to 0.72): Started OK, CUDA graph capture was in progress, got to "Ready", but then **also crashed with OOM during inference** during the draft model's `forward_draft_extend`

4. **Now**: fcloud JupyterLab API is returning 503 — instance needs restart

**Root cause of OOM during inference**: The draft model processes the FULL prefill token sequence (up to ~70K tokens for long-context tasks) during `forward_draft_extend`. For 70K tokens through the MLP, the gate_up_proj output alone is ~3 GB. Combined with target model activations, aux_hidden_states (~1.6 GB for 3 layers × 70K tokens), and existing allocations, it exceeds the ~20 GB available after pool + model weights.

**Possible fixes** (in order of preference):
1. Test with the Mamba rollback fix first — the previous OOM might have been exacerbated by corrupted recurrent state accumulating memory
2. Reduce `mem-fraction-static` further (e.g., 0.65) to leave more room for draft model forward
3. Add chunked processing in `forward_draft_extend` for long inputs

**The fcloud instance needs to be restarted.** Could you restart it so we can continue testing?
