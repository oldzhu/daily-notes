

Fetched https://soar.openbmb.cn/leaderboard

Now let me analyze the data and estimate our position. 

Created 4 todos

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Starting: *Update copilot instructions* (4/4)

Now update copilot instructions with the leaderboard checking mandate: 

Read [](file:///home/oldzhu/sglang/.github/copilot-instructions.md#68-68), lines 68 to 85

Read [](file:///home/oldzhu/sglang/.github/copilot-instructions.md#85-85), lines 85 to 110

Replacing 7 lines with 22 lines in [](file:///home/oldzhu/sglang/.github/copilot-instructions.md)


```

```


Completed: *Update copilot instructions* (4/4)

## Leaderboard Analysis (2026-04-12)

### Current Standings

| Rank | Team | Score | Gap from us |
|------|------|-------|-------------|
| 1 | 智算一队 | **100** | +76.5% |
| 2 | Slightwind | **87.68** | +54.8% |
| 3 | dwq | **86.21** | +52.2% |
| 4 | FlashSALA | **83.18** | +46.9% |
| 5 | O(N^3) Enjoyers | **79.55** | +40.5% |
| 6-8 | (3 teams) | 79.17-79.31 | ~40% |
| **19** | **team-beta** | **56.63** | — |

### Gap to Top 5

- **Need 79.55 to enter top 5** → requires **1.40× total score improvement** (40.5% gain)
- Positions #5-#8 are a **tight cluster** (79.17-79.55) — if we reach ~80, we could jump straight to #5

### Decomposing the 1.4× Improvement

Since `Score = Performance × C`:

1. **C improvement** (if current submission uses C=0.92 or 0.96):
   - 0.92 → 1.0 gives +8.7% (still need ~29% speed improvement)
   - 0.96 → 1.0 gives +4.2% (still need ~35% speed improvement)

2. **Speed improvement** (the dominant factor):
   - Need each tier Duration to drop by ~29-35% across the board
   - Current best local speeds (Test 12): S1=121.71s, S8=44.09s, Smax=35.86s

### What the Top Teams Are Likely Doing

- **#1 (score 100)**: Custom CUDA kernels, aggressive operator fusion, possibly speculative decoding. They're 77% faster than us.
- **#5 (score 79.55)**: Likely using well-optimized GPTQ + sparse/dense with kernel-level optimizations.
- **Key difference**: Top teams are likely running custom fused kernels, optimized sparse attention, and better scheduling — not just config tuning.

### Strategic Path to Top 5

Given **gap > 30%**, we need **fundamental speed improvement**:

1. **Submit Test 12 config immediately** — if current submission is older/slower, this alone may jump several ranks
2. **Kernel optimization** — custom CUDA kernels for attention, GEMM fusion
3. **Speculative decoding** — can give 1.5-2× decode speedup if done right
4. **Sparse attention (if fixable)** — sparse was 20-40% faster than dense for long context
5. **Operator fusion** — fuse normalization+projection, reduce kernel launch overhead

Updated both copilot-instructions.md (with leaderboard tracking requirement) and persistent memory ([/memories/soar_2026_leaderboard.md]()) with the full snapshot. Future sessions will automatically check the leaderboard after each submission.

Made changes.
