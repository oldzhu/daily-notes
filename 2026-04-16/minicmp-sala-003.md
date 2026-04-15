Created 7 todos

Ran terminal command:  cd /home/oldzhu/sglang && git worktree add ../sglang-eagle3 -b eagle3-spec-decode HEAD

Starting: *EAGLE3 deep research & implementation plan* (2/7)

Now let me launch the EAGLE3 research as a subagent while I work on FLA kernel analysis in parallel.  

EAGLE3 spec decode deep research

FLA SimpleGLA kernel optimization research

Searched for files matching `**/multilayer_eagleworker.py`, no matches

Searched for files matching `**/eagle_utils.py`, 2 matches

Searched for files matching `**/eagle_info.py`, 2 matches

Searched for files matching `**/multi_layer_eagle_utils.py`, 2 matches

Searched for files matching `**/llama_eagle3.py`, 2 matches

Searched for files matching `**/minicpm_sala.py`, no matches

Searched for files matching `**/hybrid_linear_attn_backend.py`, 2 matches

Searched for files matching `**/radix_linear_attention.py`, no matches

Searched for files matching `**/python/sglang/srt/layers/attention/fla/**/*.py`, 20 matches

Searched for files matching `**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, 2 matches

Searched for files matching `**/python/sglang/srt/layers/attention/**/radix_linear*.py`, no matches

Searched for text `RECURRENT_THRESHOLD`, 19 results

Searched for regex `state_pool|state_indices|fast_state_io`, 20 results

Searched for files matching `**/*eagle*.py`, 30 matches

Searched for files matching `**/models/*minicpm*.py`, 8 matches

Searched for files matching `**/speculative/**/*.py`, 30 matches

Searched for text `CHUNK_SIZE`, 20 results

Searched for text `memory_pool`, 20 results

Searched for text `state_load`, no results

Searched for text `state_store`, 20 results

Searched for regex `gather|scatter` (`**/python/sglang/srt/layers/attention/**`), 20 results

Searched for text `minicpm_sala`, 20 results

Searched for files matching `**/attention/**/*.py`, 50 matches

Searched for files matching `**/fla/**/*.py`, 20 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/chunk.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/fused_recurrent.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/op.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#300-300), lines 300 to 1480

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/chunk_delta_h.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/utils.py#1-1), lines 1 to 50

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1550-1550), lines 1550 to 1700

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1700-1700), lines 1700 to 1703

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/chunk_o.py#1-1), lines 1 to 100

Searched for text `FLA_CHUNK_SIZE`, 20 results

Searched for text `initial_state_indices`, 20 results

Searched for text `_can_use_fast_state_io` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 3 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/chunk_scaled_dot_kkt.py#1-1), lines 1 to 100

Searched for text `is_target_verify`, 30 results

Searched for text `residual_scale`, 30 results

Searched for files matching `**/models/minicpm*.py`, 8 matches

Searched for text `SimpleGLAAttnBackend`, 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1540-1540), lines 1540 to 1600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/index.py)

Searched for regex `class.*Model|SimpleGLA|simplegla` (`**/python/sglang/srt/models/*.py`), 50 results

Searched for text `force_dense_minicpm`, 20 results

Searched codebase for "memory pool state allocation SimpleGLA temporal state storage", 14 results

Searched for text `temporal` (`**/python/sglang/srt/mem_cache/**`), 30 results

Searched for text `fast_state_io` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_utils.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_info.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#270-270), lines 270 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#400-400), lines 400 to 600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/multi_layer_eagle_utils.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/llama_eagle3.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 250

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#480-480), lines 480 to 600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#980-980), lines 980 to 1100

Searched for text `class.*Worker` (`**/python/sglang/srt/speculative/**/*.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#800-800), lines 800 to 1100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1200-1200), lines 1200 to 1400

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_utils.py#200-200), lines 200 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/fused_recurrent.py#1-1), lines 1 to 500

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/multi_layer_eagle_worker.py#1-1), lines 1 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/multi_layer_eagle_worker.py#200-200), lines 200 to 753

Searched for text `class SimpleGLAAttnBackend` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#600-600), lines 600 to 1350

Searched for text `hidden_size|draft_vocab_size` (`**/python/sglang/srt/models/minicpm.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1440-1440), lines 1440 to 1700

Searched for text `hidden_size|target_hidden_size|draft_vocab_size` (`**/python/sglang/srt/models/llama_eagle3.py`), 20 results

Searched for text `MiniCPMModel` (`**/python/sglang/srt/models/minicpm.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/chunk.py)

Read memory

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/simplegla_spec_decode_compatibility_research.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/spec_decode_recurrent_eagle_research.md)

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/fla_kernels_comprehensive_analysis.md)

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_vrtx_01Lojbf2Q2n5jySpAA5FS6SZ__vscode-1776120181649/content.txt#1-1), lines 1 to 200

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_vrtx_01Lojbf2Q2n5jySpAA5FS6SZ__vscode-1776120181649/content.txt#200-200), lines 200 to 450

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_vrtx_01Lojbf2Q2n5jySpAA5FS6SZ__vscode-1776120181649/content.txt#450-450), lines 450 to 650

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_vrtx_01Lojbf2Q2n5jySpAA5FS6SZ__vscode-1776120181649/content.txt#650-650), lines 650 to 730

Excellent. Both research reports are comprehensive. Let me save the key findings and start working on the concrete proposals. 

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/eagle3_implementation_plan.md)

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/fla_kernel_optimization_findings.md)

Now let me verify the exact chunk size constant and FLA kernel locations:  

Searched for regex `CHUNK_SIZE|FLA_CHUNK_SIZE|chunk_size.*=.*64` (`**/python/sglang/srt/layers/attention/fla/**`), 20 results

Searched for text `residual_scale` (`**/python/sglang/srt/models/minicpm.py`), 3 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#580-580), lines 580 to 650

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/chunk.py#1-1), lines 1 to 45

Now I have all the context needed. Let me write up a comprehensive proposal document for all three tracks:   

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/chunk_delta_h.py#1-1), lines 1 to 30

Starting: *Write combined optimization proposal* (4/7)

Completed: *Write combined optimization proposal* (4/7)

Here's the complete plan with all three tracks:

---

## Summary of Research Findings

Git worktree created at sglang-eagle3 on branch `eagle3-spec-decode`. Both deep-dives are complete. Here's what to work on this week:

---

### Track 1: Tier B Quick Wins (main branch, config/code changes)

These are testable in 1-2 fcloud sessions:

| # | ID | Change | Where | Expected Gain | Risk |
|---|-----|--------|-------|--------------|------|
| 1 | **A4** | `RECURRENT_THRESHOLD=64` (test 64, 96, 192) | `prepare_env.sh` env var | 1-3% | Zero |
| 2 | **S4** | `--schedule-conservativeness 0.95` | `prepare_env.sh` server arg | 3-5% | Medium (OOM risk) |
| 3 | **M1** | Fold `residual_scale` into RMSNorm weights at load time | minicpm.py — eliminate 64 multiply ops per forward | 1-3% | Zero (math-equivalent) |
| 4 | **K4** | Fuse `*= residual_scale` into `fused_add_rmsnorm` kernel | sgl-kernel CUDA | 1-2% | Low |

Items 1-2 are config-only, testable immediately. Items 3-4 need code changes.

---

### Track 2: SimpleGLA/FLA Kernel Optimization (main branch)

**Key finding**: Chunk size is **hardcoded to 64** at chunk_delta_h.py. The `chunk_local_cumsum` call also hardcodes `chunk_size=64` at chunk.py.

| # | ID | Change | Impact | Effort |
|---|-----|--------|--------|--------|
| 1 | **A2** | Make chunk size configurable via env var, test 32/128 | 5-10% prefill | Low — change 2 constants + add env var |
| 2 | **A1** | State contiguity guarantee in memory pool alloc | 5-8% decode | Medium — modify `memory_pool.py` alloc logic |
| 3 | **A3** | Fuse state I/O into FLA kernel | 10-15% decode | High — requires FLA kernel modification |

**Recommended order**: A2 first (low-effort, high-impact), then A1.

For **A2**, the changes would be:
- chunk_delta_h.py: `CHUNK_SIZE = int(os.environ.get("SGLANG_FLA_CHUNK_SIZE", "64"))`
- chunk.py: pass `chunk_size=CHUNK_SIZE` instead of hardcoded 64
- Constraint: must be power of 2 (enforced by `cumsum.py#L171`)

---

### Track 3: EAGLE3 Speculative Decoding (eagle3-spec-decode branch)

**Critical findings from research:**

1. **SGLang EAGLE3 pipeline is fully built** — draft worker, tree mask, verification all exist
2. **Mamba2 already supports `is_target_verify`** in `hybrid_linear_attn_backend.py` — we have a working template
3. **SimpleGLA does NOT support it** — the `forward()` method at line 1550 has no `is_target_verify` check

**What SimpleGLA is missing** (vs Mamba2 which works):

| Feature | Mamba2 | SimpleGLA |
|---------|--------|-----------|
| Detect TARGET_VERIFY | Yes | **Missing** |
| Intermediate state cache | Yes | **Missing** |
| disable_state_update param | Yes (kernel level) | **Missing** |
| State rollback on rejection | Yes | **Missing** |

**MVP Implementation Plan** (sequential verify approach — simplest, ~10% slower verify but spec decode is still 2-3x net positive):

1. **Create `minicpm_eagle3.py`** — draft model based on `llama_eagle3.py` template (~300M params, 1 attention layer + fc + lm_head)
2. **Modify `hybrid_linear_attn_backend.py`** — add `is_target_verify` branch to SimpleGLA's `forward()`:
   - Save state before processing verify tokens
   - Process tokens sequentially (one at a time) storing intermediate states
   - Don't update global state pool during verify
3. **Add intermediate state buffer** to memory pool for SimpleGLA layers
4. **Add state restore logic** after verification determines accepted count
5. **Train draft head** — collect hidden states from SALA, distill a small head

**Biggest blocker**: Training the draft head requires SALA hidden states from real inference. We need to:
- Run inference with hidden state capture enabled
- Train the 300M param head (few hours on a single GPU)
- Package the head weights (fits in 2GB submission limit at ~600MB FP16)

---

### Recommended Weekly Plan

| Day | Track 1 (Tier B) | Track 2 (FLA) | Track 3 (EAGLE3) |
|-----|-----------------|---------------|------------------|
| Mon | A4 threshold tuning (config) | A2 chunk size env var (code) | Start draft model skeleton |
| Tue | S4 conservativeness (config) | Test chunk sizes 32/128 on fcloud | Add is_target_verify to SimpleGLA |
| Wed | M1 residual scale folding | A1 state contiguity research | Sequential verify wrapper |
| Thu | Combine all Track 1+2 wins | Test combined on fcloud | State rollback logic |
| Fri | Submit v19 package | — | Test EAGLE3 integration locally |

Want me to start with **Track 1 item A4** (recurrent threshold) + **Track 2 item A2** (chunk size env var) first? These are the lowest-risk, highest-impact changes to implement right now.
