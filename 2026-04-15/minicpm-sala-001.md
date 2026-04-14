Starting: *Remove NGRAM flags from prepare_env.sh* (1/4)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#128-128), lines 128 to 138

Replacing 1 lines with 1 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)


```

```


Ran terminal command:  cd ~/sglang && git add benchmark/soar/demo_sala/prepare_env.sh && git commit -m "Remove NGRAM spec decode flags: incompatible with SimpleGLA lightning layers

NGRAM speculative decoding requires target_verify which is incompatible
with recurrent attention (SimpleGLA). 24/32 layers are lightning/recurrent,
so verification would silently corrupt recurrent state with no rollback.
Keep --force-dense-minicpm --kv-cache-dtype fp8_e5m2 (correct baseline config)."

Ran terminal command:  cd ~/sglang && git push minicpm-src mixed_minicpm_cudagraph && git push origin mixed_minicpm_cudagraph

Starting: *Research spec decode with hybrid recurrent* (2/4)

Now let me research if spec decode can be made to work with hybrid recurrent architectures:  

Research spec decode + recurrent attention

Searched for files matching `**/python/sglang/srt/speculative/**/*.py`, 30 matches

Searched for files matching `**/python/sglang/srt/layers/attention/**/*.py`, 30 matches

Searched for regex `mamba|recurrent|linear_attn|gla|simplegla` (`**/python/sglang/srt/**`), 50 results

Searched for regex `checkpoint|rollback|snapshot|save_state|restore_state` (`**/python/sglang/srt/**`), 50 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/base_spec_worker.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 150

Searched for text `restore_state` (`**/python/sglang/srt/speculative/eagle_worker.py`), 2 results

Searched for text `target_verify` (`**/python/sglang/srt/**`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_worker.py#475-475), lines 475 to 550

Searched for text `state` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_worker.py#380-380), lines 380 to 430

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#985-985), lines 985 to 1100

Searched for text `SimpleGLA` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 9 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#500-500), lines 500 to 900

Searched for text `SpeculativeState` (`**/python/sglang/srt/mem_cache/**`), 10 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1440-1440), lines 1440 to 1703

Searched for regex `recurrent|linear_attn|state|verify|draft` (`**/python/sglang/srt/speculative/**`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/mem_cache/memory_pool.py#140-140), lines 140 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/mem_cache/memory_pool.py#200-200), lines 200 to 400

Searched for text `forward_mode` (`**/python/sglang/srt/speculative/eagle_worker.py`), 26 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_worker.py#600-600), lines 600 to 750

Searched codebase for "speculative decoding recurrent attention state rollback checkpoint", 30 results

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/simplegla_spec_decode_compatibility_research.md)

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/spec_decode_recurrent_implementation_guide.md)

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/spec_decode_recurrent_eagle_research.md)

Starting: *Update memory with NGRAM incompatibility* (4/4)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_scoring_rules.md)

Inserted into memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_scoring_rules.md)

Completed: *Update memory with NGRAM incompatibility* (4/4)

Done. Here's the summary:

## 1. NGRAM Flags Removed
Committed `290e370e6` — removed `--speculative-algorithm NGRAM --speculative-num-draft-tokens 12` from prepare_env.sh. Kept `--force-dense-minicpm --kv-cache-dtype fp8_e5m2`.

## 2. Spec Decode + Hybrid Recurrent: Research Findings

**ALL speculative decoding algorithms** (NGRAM, EAGLE, Standalone) are incompatible with SimpleGLA lightning layers because:

| Problem | Detail |
|---------|--------|
| **Recurrent state corruption** | `state[t] = f(state[t-1], k, v)` — rejected draft tokens permanently modify state with no rollback |
| **No guard code** | SimpleGLA doesn't check `is_target_verify()` — proceeds silently with wrong computation |
| **75% of layers affected** | 24/32 layers are lightning/SimpleGLA |

**Mamba2 solved this** with `SpeculativeState` (intermediate state checkpointing in memory_pool.py) + `disable_state_update=True` in the kernel. SimpleGLA lacks this infrastructure.

### Fix Options (if we want spec decode)

| Option | Effort | Speed Impact | Risk |
|--------|--------|-------------|------|
| **Sequential verify** — process 1 draft token at a time for lightning layers | ~2h | 10-20% slower on verify step | Low |
| **State checkpointing** — save state after each draft token, restore on rejection (match Mamba2 pattern) | ~8h | ~0% | Medium |
| **Do nothing** | 0 | N/A | Lose all spec decode benefit |

Memory cost for checkpointing: ~5.3 GB (64 concurrent × 8 draft tokens × 24 layers × 262KB/state).

**Recommendation for SOAR**: Implementing sequential verify (Option 1) first is the lowest-risk path, but given the competition timeline, we should first exhaust simpler speed optimizations (torch.compile, residual scale folding) before investing in spec decode support.

