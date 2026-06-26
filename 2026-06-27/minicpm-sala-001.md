I’m locking the approved change into a narrow slice now: add a small-`M` Marlin fallback plus opt-in tracing, then validate locally without widening scope.

Searched for files matching `**/docs/soar_2026_changes/CHANGE_*`, 200 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1-1), lines 1 to 120

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/conversation_tracking_rule.md)

I have the exact local edit slice: one Python branch in the GPTQ apply path, plus one env knob, with docs alongside it. Next I’m checking the conversation-tracking note and picking the doc filenames before patching.

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_003.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_003.zh.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/conversation_tracking_rule.md)

I’m making the smallest grounded code change now: one thresholded fallback and one opt-in trace in the GPTQ apply branch, plus matching env knobs and iteration docs.

Ran terminal command:  cd /home/oldzhu/sglang && date '+%Y%m%d_%H%M'

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#268-268), lines 268 to 294

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#520-520), lines 520 to 536

Generating patch (90 lines) in [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py)


```

```
The code change is in place. I’m doing the narrowest validation next: check the touched Python file for diagnostics before I widen to env/doc updates.

Checked [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py), no problems found

I’m adding the runtime knobs now so the new branch can be controlled from the official launch path and traced without hardcoding anything.

Generating patch (32 lines) in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)


```

```
The runtime knobs are wired in. Next I’m adding the required bilingual change record and chat log, then I’ll run a focused validation over the touched files.

I’m writing the mandatory bilingual records now so the code change and the reasoning stay auditable with the repo’s workflow.

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_004.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_004.zh.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-small-m-fallback-trace_20260529_0709.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-small-m-fallback-trace_20260529_0709.zh.md)

The implementation and docs are in place. I’m running the narrow local validation now: Python syntax for the touched code and shell syntax for the launch script.

Ran terminal command:  cd /home/oldzhu/sglang && python3 -m py_compile python/sglang/srt/layers/quantization/gptq.py

Ran terminal command:  cd /home/oldzhu/sglang && bash -n benchmark/soar/demo_sala/prepare_env.sh

Checked [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py), [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh), no problems found

Implemented the approved diagnostic change: W4A8 REAL now falls back to Marlin for small M under a configurable threshold, adds opt-in dispatch tracing, updates prepare_env knobs, and records the iteration in bilingual change/chat docs. Local validation passed for the touched Python and shell files; no fcloud test was run in this turn.



The approved change is in place.

In gptq.py, the W4A8 REAL apply path now does two new things:
it falls back to the existing Marlin path when `M_orig` is below `SOAR_W4A8_REAL_FALLBACK_M`, and it emits lightweight per-call logs when `SOAR_W4A8_TRACE=1` so we can see which layers used `fused` versus `fallback=marlin`, along with `M`, `M_pad`, `N`, and `K`.

In prepare_env.sh, I added the matching runtime knobs:
`SOAR_W4A8_REAL_FALLBACK_M` defaulting to `128`, and `SOAR_W4A8_TRACE` defaulting to `0`. I also added them to the startup echo block so remote runs show the active settings clearly.

I recorded the iteration in CHANGE_W4A8_001_iteration_004.en.md, CHANGE_W4A8_001_iteration_004.zh.md, and the bilingual chat logs under CHAT_w4a8-small-m-fallback-trace_20260529_0709.en.md and CHAT_w4a8-small-m-fallback-trace_20260529_0709.zh.md.

Local validation passed:
`python3 -m py_compile python/sglang/srt/layers/quantization/gptq.py`
and
`bash -n benchmark/soar/demo_sala/prepare_env.sh`

The next natural step is a focused fcloud A/B on the isolated `cwe147` repro:
1. `SOAR_W4A8_REAL_FALLBACK_M=0 SOAR_W4A8_TRACE=1`
2. `SOAR_W4A8_REAL_FALLBACK_M=128 SOAR_W4A8_TRACE=1`

If you want, I can prepare and run that remote comparison next, but I will wait for explicit approval before starting any fcloud test.

