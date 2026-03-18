I have approval. I’m inspecting the launcher path one more time so I can keep this iteration tight: add scored selection among existing Marlin variants, cache the decision per shape, and fold the runtime-vs-static discussion into a new EN/ZH change doc.

Searched for regex `cudaFuncSetAttribute|MaxDynamicSharedMemorySize|cudaOccupancyMaxActiveBlocksPerMultiprocessor|unordered_map|mutex` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), 2 results

Searched for files matching `**/docs/soar_2026_changes/CHANGE_0045*`, no matches

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#860-860), lines 860 to 1120

I found the key implementation constraint: the current launcher always asks for the full opt-in dynamic shared-memory budget, which makes occupancy scoring tricky if we try to turn it into actual multi-block scheduling. I’m checking how that value is used inside the kernel before I decide whether to keep occupancy as a heuristic only or make it affect launch behavior.

Searched for text `max_shared_mem` (`**/sgl-kernel/csrc/gemm/marlin/**`), 30 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1-1), lines 1 to 260

The kernel uses the passed shared-memory budget for its internal shared-memory layout, so I’m keeping this iteration conservative: occupancy can inform selection, but I don’t want to rewrite launch memory budgeting in the same feature. I’m reading that exact section and then I’ll patch only the scored-and-cached selector.

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#280-280), lines 280 to 380

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#380-380), lines 380 to 520

I’ve narrowed the safe scope. The change I’m making is: runtime scored selection only among already-compiled Marlin variants, plus a per-shape cache so the scoring cost is paid once per hot shape rather than on every GEMM launch.

Created 5 todos

I’m editing the Marlin launcher now. The code change is intentionally narrow: keep the existing compiled kernels, add scored selection for SM120 at runtime, and cache that decision per shape so the scoring overhead doesn’t repeat in steady state.

Generating patch (723 lines) in [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0045_sm120_marlin_scored_runtime_selection.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0045_sm120_marlin_scored_runtime_selection.zh.md)


```

```

```

```

```

```
The scored selector and docs are in. I’m validating the edited files now for obvious compiler issues before I tell you exactly what changed and any remaining risk in the implementation.

Checked [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0045_sm120_marlin_scored_runtime_selection.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0045_sm120_marlin_scored_runtime_selection.zh.md), no problems found

Ran terminal command:  cd /home/oldzhu/sglang && git diff -- sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu docs/soar_2026_changes/CHANGE_0045_sm120_marlin_scored_runtime_selection.en.md docs/soar_2026_changes/CHANGE_0045_sm120_marlin_scored_runtime_selection.zh.md | sed -n '1,260p'

Validation passed on the edited files. I’m glancing at the final diff once more for shape-cache and logging details, then I’ll summarize what changed and what still needs fcloud validation.

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_PMCS7KoxOzmbyTRV8YzKKbP5__vscode-1773788064229/content.txt#1-1), lines 1 to 260

Completed: *Implement scored config selection* (1/5)

Implemented the approved `CHANGE_0045` scope in gptq_marlin.cu. On SM120 only, the launcher now scores all valid precompiled Marlin configs instead of returning the first valid one, and caches the chosen config per shape so the host-side scoring cost is paid once per hot shape rather than on every launch. I kept non-SM120 behavior unchanged. The SM120 one-time log now also reports whether the config came from scoring or cache, plus the occupancy estimate and score.

I also added the discussion record you asked for in CHANGE_0045_sm120_marlin_scored_runtime_selection.en.md and CHANGE_0045_sm120_marlin_scored_runtime_selection.zh.md. Those docs explicitly capture:
- what “auto-scored” means
- that selection is runtime-dynamic per GEMM shape, not a one-time initialization choice
- why one static config is unlikely to be best across all shapes
- why the scoring overhead should be acceptable when amortized through the shape cache

I validated the edited files for diagnostics, but I did not build the wheel or run correctness/benchmark locally in this environment. One implementation detail to keep in mind: occupancy is currently used as a selection heuristic only; I did not change the actual launch-side shared-memory budgeting in this iteration, because that would have been a higher-risk kernel-launch behavior change.

Natural next steps:
1. Rebuild the patched sgl-kernel wheel on fcloud and verify the new SM120 log line includes `source=score` or `source=cache`.
2. Re-run correctness first, because the recent `acc_ori` is still below a safe submission threshold.
3. If correctness is stable, compare `S1`, `S8`, and `Smax` against the `CHANGE_0041` baseline and the previous `CHANGE_0043` wheel.

