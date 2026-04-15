Let me check the leaderboard and review our optimization catalog to plan next steps.  



Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#1-1), lines 1 to 150

Fetched https://soar.openbmb.cn/leaderboard

Compacted conversation

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#150-150), lines 150 to 200

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#1-1), lines 1 to 50

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#50-50), lines 50 to 83



Fetched https://soar.openbmb.cn/toolkit

Now let me update the leaderboard memory and provide the analysis. 

Read memory

Read memory [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Here's the comprehensive analysis:

---

## Current Situation

**Leaderboard (2026-04-14):**

| Rank | Team | Score |
|------|------|-------|
| 1 | FlashSALA | 99.89 |
| 2 | 智算一队 | 99.82 |
| 3 | Slightwind | 87.52 |
| 4 | dwq | 86.05 |
| **5** | **O(N^3) Enjoyers** | **79.41** |
| 6-8 | (tight cluster) | 79.03-79.16 |
| 19 | **team-beta** | **56.54** |

**Notable change**: FlashSALA jumped from #4 (83.18) → #1 (99.89) in 2 days — they clearly deployed a major breakthrough (likely EAGLE3 spec decode). Top 2 teams have "cracked the code."

**Estimated v18 score**: Based on fcloud improvement ratios (old sparse config → new dense+torch.compile config), our v18 submission should land around **~74-75**, which would move us from #19 to roughly **#9-#10**. Still ~5 points short of #5 (79.41).

---

## What To Do Next (Prioritized)

### Tier A: Game-Changers (required for top 5)

**1. EAGLE3 Speculative Decoding** — THE officially suggested path
- The toolkit page explicitly lists this as 路径二 and says SGLang already has "初步集成" in `hybrid_linear_attn_backend.py` (`is_target_verify` mode)
- Top 2 teams (99+) almost certainly use this — it gives **2-3x decode speedup**
- Main challenge: Lightning Attention (SimpleGLA) layers can't do tree-mask verification natively, but state save/restore IS possible
- Key files to study: `eagle_utils.py`, `multilayer_eagleworker.py`, `llama_eagle3.py` (template), `hybrid_linear_attn_backend.py`
- Effort: High (1-2 weeks), but it's the only path to match the elite cluster
- Requires training a draft head first (small, fits in 2GB limit)

**2. Custom SimpleGLA/FLA Kernel Optimization** — 75% of compute lives here
- The 24 SimpleGLA layers dominate forward pass time — optimizing FLA kernels has highest ROI
- `A2`: FLA chunk size tuning (currently hardcoded 64, test 32/96/128/192) — **5-10% prefill gain**
- `A3`: Fuse state I/O into FLA kernel (eliminate load/store per decode step) — **10-15% decode gain**
- `A1`: SimpleGLA state contiguity guarantee (ensure fast I/O always triggers) — **5-8% decode gain**

### Tier B: Medium Impact (each 1-5% gain, stackable)

**3. Residual Scale Folding (M1)** — fold `residual_scale` into RMSNorm weights at load time, eliminating 64 multiply kernels total. **1-3%**, zero risk.

**4. Schedule Conservativeness (S4)** — try `--schedule-conservativeness 0.95`. Config-only. **3-5%** but some OOM risk.

**5. Recurrent Threshold Tuning (A4)** — test `RECURRENT_THRESHOLD=64,96,192`. Config-only. **1-3%**.

**6. Fused RMSNorm + residual_scale Kernel (K4)** — modify the CUDA kernel to accept an extra scalar. **1-2%**.

### Tier C: Already Tested / Low Impact

- ~~S1 torch.compile~~ DONE
- ~~S2 mixed-chunk~~ tested in VarB, marginal (-1% S8 only)
- ~~S3 prefill-max-requests~~ tested in VarA, zero improvement
- ~~S5 max-running-requests~~ tested in VarA, zero improvement

---

## Recommended Strategy

Given the gap analysis, I recommend a **two-track approach**:

**Track 1 (Quick wins — this week)**: Stack Tier B config/code changes (items 3-6) to squeeze another 5-10% on top of v18. This could push us from ~74 to ~80, entering the #5-#8 cluster.

**Track 2 (Game-changer — parallel)**: Begin EAGLE3 speculative decoding research and implementation. This is the only path to top 3. The first step is understanding the existing SGLang EAGLE3 integration and how `hybrid_linear_attn_backend.py` handles verification mode for SimpleGLA layers.

Which track would you like to start with first? Or should I begin deep-diving into the EAGLE3 implementation while we wait for the v18 official score?
