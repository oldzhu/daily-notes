Created 3 todos

Searched for files matching `**/docs/soar_2026_changes/CHANGE_004*.md`, 20 matches

Searched for files matching `**/docs/soar_2026_changes/CHANGE_0056*.md`, 2 matches

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0043_sm120_marlin_kernel_submission_wheel.en.md#1-1), lines 1 to 120

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0043_sm120_marlin_kernel_submission_wheel.en.md#120-120), lines 120 to 139

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0045_sm120_marlin_scored_runtime_selection.en.md#1-1), lines 1 to 130

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0045_sm120_marlin_scored_runtime_selection.en.md#130-130), lines 130 to 177

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0044_deepgemm_pin_build_unblock.en.md#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0056_prepare_env_local_wheel_installs.en.md#1-1), lines 1 to 100

Now I have all the context. Let me document the build process and answer the question: 

Read memory

Starting: *Answer pip install -e vs whl question* (2/3)

First, let me add the build process to copilot instructions and memory: 

Read [](file:///home/oldzhu/sglang/.github/copilot-instructions.md#60-60), lines 60 to 85

Read [](file:///home/oldzhu/sglang/.github/copilot-instructions.md#88-88), lines 88 to 115

Now let me find a good place to insert the sgl-kernel build instructions. It should go after the fcloud setup section:  

Searched for text `fcloud instance setup` (`**/.github/copilot-instructions.md`), 1 result

Read [](file:///home/oldzhu/sglang/.github/copilot-instructions.md#220-220), lines 220 to 240

Replacing 3 lines with 35 lines in [](file:///home/oldzhu/sglang/.github/copilot-instructions.md)


```

```
Now add to memory:  

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/sgl_kernel_build_process.md)

Now commit the copilot instructions update:   

Ran terminal command:  cd /home/oldzhu/sglang && git add .github/copilot-instructions.md && git commit -m "Add sgl-kernel build & test instructions for fcloud" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Now let me answer the **`pip install -e` vs `.whl` question**, and provide the **CHANGE_0043/44/45/56 review**:   

Completed: *Answer pip install -e vs whl question* (2/3)

---

### `pip install -e sgl-kernel/` vs `.whl` — Your approach is better

| | `.whl` build + install (your way) | `pip install -e` (editable) |
|---|---|---|
| **Matches submission** | Yes — submission ships `.whl`, you test what you ship | No — editable install uses source tree |
| **Portability** | `.whl` can be copied anywhere | Editable tied to source path |
| **Build isolation** | Clean build in `dist/`, reproducible | Build artifacts in `build/`, can be stale |
| **Submission size** | `.whl` only (~50MB) | Would need entire sgl-kernel source (~200MB+) in submission |
| **Failure modes** | If `.whl` works, submission works | Could work in editable mode but fail as `.whl` (packaging issues) |
| **Dev iteration** | Slower (full rebuild each time) | Faster (incremental, reuses `.so` files) |

**Verdict**: `.whl` is strictly better for competition. The only advantage of `pip install -e` is faster dev iteration during debugging, but it can mask real packaging issues. You're right to use `.whl`.

Also note: even if you copied sgl-kernel source to `/root/submission_sim`, `pip install -e` would still trigger a full CUDA compilation (~20-40 min) since there are no cached build artifacts there, so there's no time saving either.

---

### Review of CHANGE_0043/44/45/56 — SM120 Marlin History

Here's what we did before and what's relevant to our new CHANGE_0125:

**CHANGE_0043** (SM120 Marlin kernel submission wheel):
- **What it did**: Added SM120-specific candidate thread-config tables (`small_batch_thread_configs_sm120[]`, `large_batch_thread_configs_sm120[]`) and SM120 GPU detection. Added one-time log line.
- **Result**: SM120 auto-config path was confirmed active via log, but **no meaningful speed improvement** — because it still used **first-valid** selection, not scored selection.
- **Insight for CHANGE_0125**: The SM120 config tables are already in place. Our new change adds the *missing kernel instantiations* that those tables reference.

**CHANGE_0044** (DeepGEMM pin build unblock):
- **What it did**: Fixed a dead git pin in CMakeLists.txt that blocked wheel builds on fcloud.
- **Insight**: Build infrastructure fix. The pin has been updated since. No functional relevance to CHANGE_0125 but confirms the build process works.

**CHANGE_0045** (SM120 scored runtime selection):
- **What it did**: Replaced the "first valid config" selection with **scored runtime selection** — evaluating all valid SM120 candidates per shape, scoring by fill ratio, wave ratio, M coverage, shared memory fit, and occupancy. Added per-shape caching.
- **Result**: This is the `score_sm120_candidate()` + `determine_exec_config()` + cache that **we are now relying on**. The scoring logic is already in our codebase.
- **Critical insight**: CHANGE_0045 added the scoring logic BUT it could only score configs that had kernel instantiations. **CHANGE_0125 now completes the picture** by adding the missing instantiations that CHANGE_0045's scorer needs to select from.

**CHANGE_0056** (Local wheel installs in prepare_env.sh):
- **What it did**: Converted all dependency installs (gptqmodel, transformers, torchao, sgl-kernel) to local `.whl` files in `prepare_env.sh`.
- **Insight**: Confirms the `.whl` install workflow is the standard. Our sgl-kernel rebuild process follows this same pattern.

### Key Takeaway

The SM120 Marlin pipeline was built in stages:
1. **CHANGE_0043**: Added SM120 config tables + GPU detection ✅
2. **CHANGE_0044**: Fixed build infrastructure ✅
3. **CHANGE_0045**: Added scored runtime selection ✅
4. **CHANGE_0125** (now): **Adds the missing kernel template instantiations** so the scorer can actually select the top-priority configs ← THIS IS THE MISSING PIECE

The scoring logic from CHANGE_0045 has been trying to select `{128,256,256}` but couldn't because `get_marlin_kernel()` returned `MarlinDefault` (= "not instantiated") for `(N_blocks=16, K_blocks=8)`. Our change fixes exactly that gap. 



