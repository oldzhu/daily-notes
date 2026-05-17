Accuracy run is going but extremely slow — NVFP4 path is much slower than baseline. Let me poll for completion:      

Checked terminal output

Ran terminal command:  tail -100 /home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01XhFygLPDqiMLqiGgB6Hfdv__vscode-1777849647694/content.txt 2>&1 | tail -60

Ran terminal command:  tail -c 8000 /home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01XhFygLPDqiMLqiGgB6Hfdv__vscode-1777849647694/content.txt | tr '\r' '\n' | tail -40

The eval client hit the 3600s timeout client-side at 62/150 (~41%). The NVFP4 path runs but is far too slow for the budget. Let me stop the eval, snapshot what we have, and pause the instance per cost-saving rule:       

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "pkill -f 'eval_model_001.py' 2>/dev/null; sleep 2; echo eval-killed; ls /root/data/outputs/ | tail -3" 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 60 "ls /root/data/outputs/20260504_105130/; echo ---; wc -l /root/data/outputs/20260504_105130/predictions.jsonl 2>/dev/null; echo ---summary---; cat /root/data/outputs/20260504_105130/summary*.json 2>/dev/null || echo no_summary" 2>&1 | tail -20

No partial output written (eval writes only at end). Run a quick speed-only single-request benchmark to get latency signal before pausing:       

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 180 "curl -s -X POST http://127.0.0.1:30000/generate -H 'Content-Type: application/json' -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":64,"temperature":0.0}}' | head -c 800; echo" 2>&1 | tail -10

Server outputs "Paris" but then degenerates — uniform NVFP4 weights cause severe quality loss. Latency is fine for short prompts (~57 tok/s decode); the eval slowness comes from long-context. Let me pause and document:        

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "pkill -f sglang.launch_server 2>/dev/null; sleep 3; pgrep -af sglang.launch_server || echo killed" 2>&1 | tail -5 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "pkill -9 -f sglang.launch_server 2>/dev/null; sleep 3; pgrep -af sglang.launch_server || echo killed" 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Instance paused. Now write findings doc:         

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.zh.md && git commit -m "phase-A: document first NVFP4 end-to-end attempt findings

- modelopt 0.31.0 incompat with torch 2.9 (torch.onnx._type_utils removed)
- modelopt 0.43.0 + modelopt-core 0.33.1 is verified-working combo
- prepare_env.sh + sglang modelopt_fp4 loader integration: WORKS
- Uniform NVFP4 (no FourOverSix): output collapses on long generation
- Long-context latency far worse than GPTQ baseline (likely BF16 dequant fallback)
- TODO Phase B: implement FourOverSix adaptive M=6/M=4 + investigate FP4 kernel path" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -8

Phase A end-to-end test summary:

**Plumbing works:** `SOAR_QUANT_PROFILE=nvfp4` switch, modelopt install (after pinning to 0.43.0 + modelopt-core 0.33.1 for torch 2.9 compat), and sglang's `modelopt_fp4` loader path all proven on fcloud. Server loads weights (6.65 GB), captures CUDA graphs, responds to short prompts at ~57 tok/s.

**But uniform NVFP4 fails on quality:** sanity prompt "The capital of France is" → "Paris" then degenerates into garbage. 150-sample accuracy eval timed out at ~62/150 (long-context items >5 min each). This matches the champion blog's explicit warning that **FourOverSix per-block adaptive scaling is required** — uniform NVFP4 collapses.

**Key dependency finding committed to prepare_env.sh:** modelopt 0.31.0 imports `torch.onnx._type_utils` which torch 2.8+ removed; need modelopt **0.43.0** + modelopt-core **0.33.1** for torch 2.9. Plus `rich` and `torchprofile` need explicit `--no-deps` install.

**Cleanup:** server killed, fcloud instance paused (cost-saving rule). All findings documented in CHANGE_0150_phase_a_nvfp4_baseline.en.md / .zh.md, commits `aa1304292` (Phase A wiring) → `0a56da668` (modelopt 0.43 fix) → `92cb3f086` (findings doc) pushed to `minicpm-src/mixed_minicpm_cudagraph`.

**Two paths forward (need user decision):**
1. **Phase B — FourOverSix:** add adaptive M=6/M=4 per-block scaling to recover quality.
2. **Investigate FP4 kernel path:** confirm whether sglang's `modelopt_fp4` actually uses SM120 FP4 tensor cores or falls back to BF16 dequant (likely cause of long-context slowness). 



