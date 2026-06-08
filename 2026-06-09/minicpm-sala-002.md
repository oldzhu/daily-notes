Great questions — let me clarify both.

## Question 1: Why "5× weight traffic" and not 2×?

The confusion is that I was comparing against the **INT4 Marlin baseline** (0.5 bytes/elem), not BF16 (2 bytes/elem). Let me break it down precisely:

### Weight traffic comparison for an MLP gate_up_proj layer (K=4096, N=32768 = 128M elements)

| Stage | Marlin W4A16 (baseline) | Naive W4A8 (dequant-to-temp) |
|---|---|---|
| Weight read from HBM | INT4 = **64 MB** | INT4 = **64 MB** |
| Temp FP8 write to HBM | — (no temp) | FP8 = **128 MB** ← extra! |
| GEMM reads weight from HBM | (fused, no re-read) | FP8 = **128 MB** ← re-read! |
| **Total weight HBM traffic** | **64 MB** | **320 MB** |

**320 ÷ 64 = 5×.** That's the weight-side penalty of the naive approach.

The activation side (BF16→FP8) saves traffic regardless — but it can't compensate for a 5× weight traffic explosion.

You're right that FP8 vs BF16 is 2× smaller. The 5× isn't about FP8 vs BF16 — it's about the **round-trip**: INT4→FP8 write + FP8 read adds 4 extra "weight-equivalent" passes that Marlin avoids by fusing the dequant into registers.

## Question 2: SMEM cost — would INT4→FP8 unpacking double SMEM?

**Yes, a naive unpack would double SMEM, but we can avoid that through pipelining.**

### The naive approach (bad):
```
SMEM at time T:
  [INT4 tile: 8 KB] [FP8 tile: 16 KB] → 24 KB total (3× INT4 size)
```

### The pipelined approach (correct):
```
Stage 1: Load INT4 tile_0 into SMEM buf_A  (8 KB)
Stage 2: Unpack tile_0 → FP8 into SMEM buf_B (16 KB), 
         simultaneously load INT4 tile_1 into SMEM buf_A (8 KB)
Stage 3: Feed tile_0 FP8 to QMMA, 
         unpack tile_1 → FP8 into buf_B
         load INT4 tile_2 into buf_A
...
```

Peak SMEM = `max(buf_A, buf_B) + double-buffer` ≈ **~24 KB** for two 128×128 tiles in flight.

For context: SM120 has **~100 KB of SMEM per SM** (same as Hopper). The CUTLASS SM120 FP8 kernel already uses ~40-60 KB for activation + weight tiles + scales + epilogue. Adding 8-16 KB for unpacked weights is within budget.

### And the trade is favorable:

| Resource | Cost of fused approach | Benefit |
|---|---|---|
| SMEM | +8-16 KB for unpack buffer | Saves 128 MB HBM write + 128 MB HBM read per MLP layer |
| Compute | ~1-2% overhead (simple bit-shift + lookup) | Gains 2× GEMM throughput (296 vs 148 TF) |
| HBM bandwidth | Same INT4 read as Marlin | Saves 50% activation traffic (FP8 vs BF16) |

SMEM is ~10× higher bandwidth than HBM — burning a bit more SMEM to avoid HBM round-trips is almost always a net win.

---

Shall I proceed with **Day 1** — the Python-side FP8 activation quantizer and new env gate (`SOAR_W4A8_REAL_FP8_GEMM`)?
