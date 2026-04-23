

Fetched https://soar.openbmb.cn/leaderboard

Read memory [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Now let me update the leaderboard memory and provide the analysis: 

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Now update the key deadlines and strategy sections:  

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Now update the gap analysis and score tiers:  

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Now let me check our optimization catalog to build the strategic plan:   

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#80-80), lines 80 to 140

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#140-140), lines 140 to 180

Now I have the full picture. Let me provide the comprehensive analysis:

---

## Updated Leaderboard Analysis (Post-Rerun, 2026-04-20)

### New Rankings

| Rank | Team | Score | Prev | Δ |
|------|------|-------|------|---|
| 1 | FlashSALA | **99.93** | 94.29 | +5.64 |
| 2 | Slightwind | **81.56** | ~79.22 | +2.34 |
| 3 | dwq | **77.53** | ~77.85 | -0.32 |
| 4 | 智算一队 | **67.86** | 90.22 | **-22.36** |
| 5 | Six-Seven | **64.58** | — | new |
| 6 | 香草小张 | 60.48 | | |
| 7 | O(N^3) Enjoyers | 58.13 | 71.79 | -13.66 |
| 8 | EnjoyNP | 54.1 | 71.57 | -17.47 |
| **19** | **team-beta** | **40.23** | 51.08 | **-10.85** |

### Gap to Targets
- **Top 5** (64.58): need **+24.35 pts** → **~38% faster** overall
- **Top 3** (77.53): need **+37.30 pts** → **~48% faster** overall

### Major Takeaways

1. **The new dataset completely reshuffled the board.** 智算一队 (semifinal champion) dropped 22 points from #2→#4. Many teams dropped 13-19 points. The new 68% long-context (32K-512K input) dataset **massively rewards prefill speed**.

2. **FlashSALA GAINED 5.64 points** — they clearly optimized for long-context prefill. They're nearly perfect (99.93).

3. **Our drop (-10.85) was less than many mid-tier teams** (-13 to -19), suggesting our config isn't the worst at long-context, but still needs significant improvement.

4. **Phase B calibration tuning (damp_percent/mse) is now LOW priority.** Calibration tuning improves accuracy (C coefficient) but doesn't improve speed. With the new prefill-heavy dataset, **speed is the overwhelming bottleneck**. Even going from C=0.92 to C=1.0 is only +8.7% — we need 38-48% to reach top 5/3.

### Strategic Reassessment

**Old plan** (Phase B calibration tuning): would recover at most +8.7% from accuracy coefficient. **Insufficient.**

**New priority order** (aligned with prefill-dominant dataset):

| Priority | Optimization | Expected Impact | Effort |
|----------|-------------|----------------|--------|
| **P1** | `--enable-mixed-chunk` (S2) | 3-5% | Config only |
| **P2** | `--prefill-max-requests 2-4` (S3) | 5-10% | Config only |
| **P3** | `--schedule-conservativeness 0.95` (S4) | 3-5% | Config only |
| **P4** | NVFP4 quantization | 15-30% prefill (FP4 tensor cores!) | Medium |
| **P5** | Chunked prefill tuning for 256K-512K | 5-10% | Medium |
| **P6** | SimpleGLA state contiguity (A1) | 5-8% decode | Medium |
| **P7** | Fuse state I/O into FLA kernel (A3) | 10-15% decode | High |

**P1-P3 are all config-only** and can be tested immediately on fcloud — estimated combined gain 10-20%.

**P4 (NVFP4)** is the biggest lever. On Blackwell GPUs, NVFP4 uses native FP4×FP4 tensor cores for weight GEMM, which is **2× faster than INT4 GPTQ Marlin** for prefill-dominated workloads. With 68% of inputs being 32K-512K, prefill GEMM time dominates. Slightwind (曹议, #2) already uses NVFP4 — this is likely why he gained points while others dropped. However, NVFP4 quantization requires significant accuracy validation and the GPTQModel pipeline may need adaptation.

**P5 (Chunked prefill)** matters because 512K inputs with 8192-token chunks will take many scheduling rounds. Tuning chunk size and scheduling aggressiveness directly impacts long-context latency.

### Realistic Target

With P1-P3 stacked (config-only, ~15% gain): score ~46 → still #15-16
With P1-P3 + P4 NVFP4 (if +30% prefill): score ~55-60 → approaching top 8-10
With all P1-P7: score ~60-70 → top 5-7 range

**Bottom line**: To reach top 5, we almost certainly need NVFP4. Config tuning alone won't bridge a 24-point gap.

### Recommended Immediate Actions

1. **Start fcloud → test P1-P3 config changes** (can be done in ~2 hours total)
2. **Research NVFP4 quantization pipeline for MiniCPM-SALA** — this is the high-impact project
3. **Deprioritize Phase B calibration tuning** — accuracy improvement gives <9% gain vs 38% needed
4. Keep `SOAR_GPTQ_FORCE_DENSE=1` and the Phase B env vars in code (no cost), but don't spend fcloud time on re-quantization experiments

Shall I start researching the NVFP4 path for MiniCPM-SALA, or would you prefer to test P1-P3 config changes on fcloud first? 


