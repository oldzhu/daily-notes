



Fetched 3 resources

Fetched https://soar.openbmb.cn/leaderboard

## Current official standings (as of 2026-04-24 18:29)

**team-beta: No.20, score 39.25** (same as pre-v18 resub). Latest three official results aligned with our local findings:

| Submission | acc_ori | C | S1 | S8 | Smax | final_score | Notes |
|---|---|---|---|---|---|---|---|
| **pre-v18 resub** | 79.24 | **1.0** | 596.25 | 1066.35 | **2746.62** | **39.25** | Current leaderboard best |
| **v18 resub** | 80.13 | **1.0** | 586.64 | 1087.90 | 2857.64 | 39.01 | Tiny S1 win, Smax loss |
| **v19 resub** | 78.07 | **0.92** | 591.04 | 1113.65 | 2917.74 | 35.37 | **Worst in both dims** |

**Confirms local findings**: v19 is officially slower AND less accurate. Don't ship v19 ever.

**Gap to top 5** (Slightwind #5 = 79.52): need **2.03×** our current score. Target cut durations by ~50%.

**Current top 5**:
1. **FlashSALA — 97.47** (Week 6 champion, Marlin tile tuning blog just released today)
2. dwq — 82.61
3. 香草小张 — 81.09
4. Slightwind — 79.52
5. 智算一队 — 79.52 (account retired per Week 6 post — team name change)

---

## Champion playbook (distilled from Weeks 4/5/6)

| Week | Topic | Technique | Expected gain | We have? |
|---|---|---|---|---|
| 4 | **Infrastructure + W4A16** | Local eval set matching **official speed-set length distribution**; W8A8 useless, W8A16 tiny, **W4A16 GPTQ+Marlin is the sweet spot**; calibration dataset composition matters | Foundation | ✅ GPTQ+Marlin ❌ our local speed set was hand-built, does NOT match official long-context |
| 4 | Operator fusion | RMSNorm+RoPE fused | ~3-5% | ✅ already on |
| 5 | **Mixed-precision KV** | Layer-wise sensitivity ablation → **first & last layers FP8, middle layers NVFP4 (FP4 E2M1)**; halves KV bandwidth for most of the model; pure FP4 drops acc to 75% | **20-30% decode speedup on long context** | ❌ we use FP8 e5m2 everywhere |
| 6 | **Marlin tile sweep** | Multi-tier `determine_exec_config`: per-M×N tile tables, decode favors K-direction unroll, prefill favors N-direction; SMEM/divisibility checked per config | **5-15% prefill+decode** | ❌ CHANGE_0125 only added SM120 tile *instantiations* — the dispatch logic still picks defaults |
| 6 | **Marlin atomic_add fix** | Original: `ceil(M/64)·N ≤ 16384` → atomic_add, else barrier; small-M was hitting the slow barrier path. Fix: always atomic_add for small M | **2-5% decode** (one-line patch) | ❌ easy free lunch |

**The trajectory is clear**: all three champions at the top stack (W4A16 Marlin) + (mixed-precision KV) + (per-shape Marlin tuning). We have only step 1.

---

## Proposed next-action plan (3 parallel tracks)

### **Iteration M1 — Marlin per-shape tile dispatch + atomic_add** (Week 6 champion's recipe)
- **Risk**: LOW (numerics unchanged, config-only kernel selection)
- **Effort**: 2-3 days
- **Expected gain**: +5-15% on S1/S8 (GEMM ≈ 85% of prefill per our profile `docs/soar_2026_changes/CHANGE_0120_profiling_analysis.md`)
- **Concrete steps**:
  1. Catalog MiniCPM-SALA linear-layer shapes (N values per layer: qkv_proj, o_proj, gate_up_proj, down_proj, lm_head)
  2. Modify `sgl-kernel/csrc/gemm/gptq_marlin/gptq_marlin.cu::determine_exec_config` — add per-M×N tile preference tables
  3. Rebuild wheel (incremental, ~5 min with ccache preserved)
  4. Apply atomic_add patch: remove `ceil(M/64)·N ≤ 16384` guard for small M (< 32)
- **Validation**: Test 12 reference S1=121.71s / S8=44.09s / Smax=35.86s. Success = any improvement without accuracy regression.
- **Why first**: additive with everything else; fastest feedback loop; already have Marlin-SM120 tile table from CHANGE_0125

### **Iteration M2 — Mixed-precision KV cache (FP4 middle layers + FP8 edge layers)** (Week 5 champion's semifinal recipe)
- **Risk**: MEDIUM (requires NVFP4 KV adapter in SGLang — non-trivial); accuracy sensitive to layer selection
- **Effort**: 1-2 weeks (most heavy item)
- **Expected gain**: **+20-30% on Smax** — this is where our official durations are worst (Smax=2746 vs S1=596); long context is memory-bound
- **Concrete steps**:
  1. Per-layer sensitivity ablation harness: force each layer's KV to FP4 one at a time on public set, measure acc drop
  2. Implement NVFP4 KV memory pool in mem_cache (FP4 E2M1, block scale factor, uint8 packed)
  3. Adapt attention backend (`srt/layers/attention/minicpm_flashinfer.py`) to support mixed-dtype KV
  4. `prepare_env.sh` env var `SGLANG_KV_LAYER_DTYPE_JSON` for per-layer dtype selection
- **Risk mitigations**: keep FP8 fallback; Test 21 (pure NVFP4 W4A4) showed catastrophe → KV-only is safer; reuse torchao or sgl-kernel NVFP4 primitives if available.
- **Why second**: biggest expected delta for Smax, but highest effort. Parallelize with M1.

### **Iteration M3 — Official-aligned local speed benchmark set**
- **Risk**: zero (no code change to submission)
- **Effort**: 1 day
- **Expected gain**: not direct, but unblocks all other work. Currently our local speed set gives 110s/40s/35s while official gives 590/1080/2850s — **ratios are inverted**, so local optimization ≠ official gain.
- **Concrete steps**:
  1. Analyze official speed duration ratios: S1 596s / S8 1066s / Smax 2746s → the **S8 > S1 pattern** tells us official S8 is long-context-heavy. Local speed_s8.jsonl must be lengthened accordingly.
  2. Reverse-engineer input/output length distribution from `benchmark_duration` × TPS assumptions
  3. Build `speed_s1_v2.jsonl` / `speed_s8_v2.jsonl` / `speed_smax_v2.jsonl` with long-context samples (64-128K inputs, 10-30K outputs)
  4. Add to demo_sala and update docs

### **Iteration M4 (stretch) — MTP/EAGLE3 with proper draft training**
- **Risk**: HIGH (Test 22 EAGLE3 untrained draft gave C=0 at 74.33% acc, 65% slower S1)
- **Effort**: 2+ weeks, needs GPU hours to train draft head
- **Expected gain**: 1.5-2× decode if accept rate > 0.5
- **Punt until after M1/M2 land** — MiniCPM-SALA may already ship an MTP layer; check the 90-ckpt first. If not, skip.

---

## Recommended sequencing

```
Week 1 (days 1-3):  M1 Marlin tile dispatch + atomic_add      → ship v20
Week 1 (days 4-5):  M3 official-aligned speed benchmark       → unblock iteration
Week 2 (days 1-7):  M2 mixed-precision KV — ablation + impl   → ship v21
Week 3+ (stretch):  M4 if MTP layer exists                    → ship v22
```

**Break-even to #5**: M1 alone probably gets us to ~42-45 (rank ~15). M1+M2 together is the plausible path to ~70-80 (top 5). M1+M2+M4 is the plausible path to top 3.

**Critical guardrails** (preserve from history):
- **Keep v18 as baseline** for all iterations (GPTQ + FP8 KV + dense + torch.compile max-bs=8 + `--skip-server-warmup`-off)
- **Never push to origin, only minicpm-src**
- **Never modify `eval_model_001.py`**
- **Always shut down fcloud** after each test round

---

## Proposed immediate next action

Start with **M1 (Marlin per-shape tile dispatch + atomic_add)** — it's the fastest to validate, matches the current #1 team's recipe from yesterday, carries zero accuracy risk, and is fully additive with M2. 

Want me to draft the full `PROPOSAL_iteration_M1_marlin_per_shape.{en,zh}.md` with concrete file changes, M×N shape catalog, and test commands for your approval?  

Read memory [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)


