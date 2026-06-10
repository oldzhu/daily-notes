Good question. Let me be precise about what we built vs what's possible.

## What we built (standalone dequant)

```
Forward pass for one MLP layer:
┌─────────────────────────────────────────────────┐
│ 1. CUDA kernel: INT4 weights → FP8 weights     │  ← gptq_int4_to_fp8_blockwise
│    Reads INT4 from HBM (64 MB)                  │
│    Writes FP8 to HBM (128 MB)                   │  ← TEMP TENSOR
│                                                 │
│ 2. CUTLASS kernel: FP8 × FP8 QMMA              │  ← existing SM120 kernel
│    Reads FP8 weights from HBM (128 MB)          │  ← RE-READS same data
│    Reads FP8 activations from HBM               │
│    Writes BF16 output                           │
└─────────────────────────────────────────────────┘
Total weight HBM traffic: 64 + 128 + 128 = 320 MB
```

## What fused would look like

```
Forward pass for one MLP layer:
┌─────────────────────────────────────────────────┐
│ Fused kernel: INT4 weights + FP8 activations    │
│    Loads INT4 weights → SMEM (64 MB from HBM)   │
│    Unpacks INT4→FP8 in SMEM (0 HBM traffic)     │  ← NO TEMP TENSOR
│    Loads FP8 activations → SMEM                 │
│    QMMA on FP8 values from SMEM                 │
│    Writes BF16 output                           │
└─────────────────────────────────────────────────┘
Total weight HBM traffic: 64 MB (same as Marlin!)
```

## Would accuracy change? **No.**

Both paths compute the **exact same** FP8 weight values:

```
INT4 value → (INT4 - zero_point) × group_scale → BF16 → FP8 e4m3
```

The standalone kernel does this in a CUDA grid, writes to HBM, then the GEMM reads it. The fused kernel does this in SMEM and feeds directly to QMMA. The math is identical — same rounding, same saturation, same FP8 values. **Accuracy would be byte-for-byte identical.**

## Would speed improve? **Yes, for S1 (decode).**

| Tier | Bottleneck | Standalone cost | Fused benefit |
|---|---:|---:|---:|
| **S1** (decode, M=1) | Weight HBM bandwidth | Extra 256 MB HBM traffic per MLP layer (write+read) | **Eliminate ~256 MB HBM traffic** → ~5-10% faster |
| **S8** (mixed) | Mixed | Extra traffic + dequant kernel overhead | Modest 2-5% |
| **Smax** (prefill, large M) | GEMM compute | Dequant cost amortized over many tokens | Marginal 0-3% |

For **S1**, the temp-FP8 tensor write+read is ~2× the actual weight data that Marlin reads. Eliminating it would bring W4A8's S1 from 110.79s down toward ~100-105s — a ~5-10% further improvement.

## Why didn't we build the fused version?

The fused kernel requires modifying the CUTLASS 3.x `CollectiveMainloop` to accept INT4 type for the B matrix and add a custom dequant transform. This is deep CUTLASS template work — the existing SM120 kernel is 380 lines of template-heavy code. Our standalone approach delivered 90% of the benefit with 10% of the effort.

The fused version is the logical **next optimization** — it's a pure speed play with zero accuracy risk.
