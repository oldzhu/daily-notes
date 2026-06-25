I’m tracing the exact dispatch path for fused W4A8 versus the baseline Marlin path so I can point to where latency could have shifted, not just speculate from the benchmark result.

Searched for regex `SOAR_W4A8_REAL_FP8_GEMM|SOAR_W4A8_FP8_GEMM|w4a8_fp8_qmma|gptq_marlin|Marlin` (`**/python/**`), 200 results

Searched for regex `SOAR_W4A8_REAL_FP8_GEMM|SOAR_W4A8_FP8_GEMM|w4a8_fp8_qmma|gptq_marlin|Marlin` (`**/sgl-kernel/**`), 200 results

Searched for regex `SOAR_W4A8_REAL_FP8_GEMM|SOAR_W4A8_FP8_GEMM|dense-as-sparse|force-dense-minicpm|kv-cache-dtype|quantization` (`**/benchmark/soar/demo_sala/**`), 200 results

I’ve found the dispatch surfaces. Next I’m reading the MiniCPM tagging logic and the two GEMM apply paths to see exactly what changes when fused W4A8 is enabled.

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#200-200), lines 200 to 320

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#800-800), lines 800 to 940

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md)

Read [](file:///home/oldzhu/sglang/sgl-kernel/python/sgl_kernel/gemm.py#470-470), lines 470 to 565

I’ve confirmed the eligibility/tagging layer and that only standard attention QKV/O are switched over. Now I’m reading the actual `apply()` branch and the fused kernel entry point, because that’s where a decode-time bottleneck would show up.

Searched for regex `_soar_w4a8_real_active|w4a8_real|w4a8_fp8_fused|apply\(` (`**/python/sglang/srt/layers/quantization/gptq.py`), 14 results

Searched for regex `w4a8_fp8_fused|w4a8_fp8_qmma|w4a8|SOAR W4A8` (`**/sgl-kernel/**`), 120 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#720-720), lines 720 to 890

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#941-941), lines 941 to 1045

The main behavioral delta is now clear: W4A8 pads every activation batch to `M=128`, converts BF16 activations to FP8 on the fly, and calls a standalone `w4a8_fused` library; the baseline stays in the in-tree Marlin path without that padding/conversion step. I’m checking the Marlin apply function and the exact tagged layers to see whether decode (`M≈1`) is paying a structural penalty.

Searched for regex `def apply_gptq_marlin_linear|apply_gptq_marlin_linear\(` (`**/python/sglang/srt/layers/quantization/marlin_utils.py`), 2 results

Searched for text `_soar_w4a8_eligible = True` (`**/python/sglang/srt/models/minicpm.py`), 4 results

Searched for regex `use_output_gate|o_gate =|_soar_w4a8_eligible` (`**/python/sglang/srt/models/minicpm.py`), 36 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#350-350), lines 350 to 520

I’ve got a falsifiable local hypothesis now: the fused path is slow on long-context decode because it pads every decode GEMM from `M=1` to `M=128`, and it enables that path not just for attention QKV/O but also MLP `gate_up/down`, multiplying the wasted work across each generated token. I’m reading the MLP tagging block to confirm the full blast radius.

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#130-130), lines 130 to 180

The blast radius is confirmed: fused W4A8 is active on QKV, attention output, MLP gate-up, and MLP down-proj. For decode that means a lot of `M=1` GEMMs are being padded to `M=128`, which is a concrete mechanism for the slowdown you’re seeing. I’m checking the runtime toggles next so I can turn this into a precise comparison/debug proposal rather than a generic suspicion.

Searched for regex `SOAR_W4A8_REAL_FP8_GEMM|SOAR_W4A8_FP8_GEMM|SOAR_W4A8_FUSED_SO` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 10 results

Searched for regex `SOAR_W4A8_REAL_FP8_GEMM|SOAR_W4A8_FP8_GEMM|_soar_w4a8_real_active|w4a8_fused` (`**/benchmark/soar/demo_sala/**`), 34 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#1-1), lines 1 to 260

I have enough to make a concrete, testable proposal. Before I wrap it up, I’m checking repo memory so I can record the key lesson if it’s not already noted.

Read memory

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/w4a8_fp8_blockwise_abandoned.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/w4a8_fp8_blockwise_abandoned.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/w4a8_fp8_blockwise_abandoned.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/dead_ends.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/dead_ends.md)

I’m saving the key structural lesson in repo memory so we don’t repeat the same “296 TFLOPS should win” assumption on small-`M` decode.

Inserted into memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/dead_ends.md)

The code comparison points to a concrete bottleneck, and it matches your suspicion that the slowdown comes from the fused W4A8 path rather than from `cwe` itself.

The key difference is not just “W4A8 vs Marlin.” The current fused path in gptq.py pads every call to `M=128`, casts activations to FP8, and then calls the standalone `w4a8_fused` kernel. The kernel itself hard-requires `M` to be a multiple of 128 in w4a8_fp8_qmma.cu. The baseline Marlin path in marlin_utils.py uses the real `M` directly, with one-time repacked weights and autotuned execution configs. On MiniCPM, fused W4A8 is enabled not only for attention QKV/O but also for MLP gate-up/down in minicpm.py and minicpm.py, so decode pays that penalty across a lot of linears per generated token.

That is exactly the kind of regime where the SM120 “296 TFLOPS FP8” headline does not help. The hardware note in SM120_RTX_PRO_HARDWARE.md already says decode at small `M` is memory-bound, not compute-bound. Here we are making it worse by turning `M=1` decode into `M=128` work. For the isolated `cwe` case, the measured behavior already fits the timeout: prefill is about `66s`, decode is about `4.48 tok/s`, and with about `1237` output tokens that is roughly $66 + 1237/4.48 \approx 342s$, which is above the `300s` wrapper timeout.

My proposal for the next step is:

1. Objective: prove or disprove that `M` padding plus broad layer tagging is the dominant cause of the W4A8 slowdown on long-output decode.
2. Expected gain: if confirmed, a small-`M` fallback to Marlin should recover most of the `cwe` slowdown while keeping the fused path available for larger prefill-style `M`.
3. Rule compliance: this stays entirely on the server/model side, leaves the eval harness untouched, and any runtime knob should live in prepare_env.sh.
4. Exact files/functions to change if approved: gptq.py for a thresholded fallback and lightweight tracing, prepare_env.sh for an env knob, and only if needed minicpm.py to narrow eligibility.
5. Cheapest discriminating check: add a temporary guard such as “if `M_orig < 128`, use Marlin instead of fused W4A8,” then rerun the isolated `cwe147` sample. If latency collapses toward the Marlin baseline, the hypothesis is confirmed immediately.
6. Follow-up trace if needed: log a histogram of `M_orig`, layer prefix, and fused-vs-Marlin dispatch counts during the isolated `cwe` run. That will tell us whether the main waste is attention, MLP, or both.

I have not changed source code yet. If you approve, the minimal next move is: add the small-`M` fallback plus per-layer dispatch tracing, then do one fcloud A/B on the isolated `cwe147` case with the new thresholded path.
