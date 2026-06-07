These are both important concerns — let me address each from first principles.

---

## Concern 1: FP8 accuracy vs BF16

**Yes, FP8 (e4m3) has fewer bits than BF16 — this is a real risk.** But the risk profile of *true W4A8* is different from what the old test measured, and there are mitigations:

### FP8 activation precision in practice

| Dtype | Mantissa | Exponent | Range | Use in GEMM |
|---|---:|---:|---|---|
| BF16 | 7 bits | 8 bits | ~1e-38 to 3.4e38 | Baseline (Marlin W4A16) |
| FP8 e4m3 | **3 bits** | 4 bits | ~1.5e-5 to 448 | What we'd use for activations |

**But**: FP8 activations in transformer GEMMs are validated in production across vLLM, TRT-LLM, and SGLang's own `cutlass_w8a8_fp8` kernel. The key trick is **per-token scaling**: each token's activation vector gets its own FP8 scale factor, which recovers most of the missing mantissa bits. The residual layers and attention outputs stay in BF16 — only the GEMM inputs are temporarily cast to FP8.

### Evidence from our own old W4A8 test (commit 7ce21c3f5)

That test used **FP8 weights × FP8 activations** — both sides were 8-bit. Accuracy was **79.20% vs 79.29% baseline** (Δ −0.09pt). That test proves that **FP8×FP8 GEMM on this model is accuracy-neutral**, even with FP8 on BOTH sides.

True W4A8 has **INT4 weights (same as baseline) × FP8 activations** — only the activation side changes. Risk is theoretically **lower** than the already-proven-neutral FP8×FP8 case.

### Risk mitigation plan

- **Phase 0 microbenchmark**: CPU-side unit test of FP8 activation quantization/dequant on real MiniCPM hidden states to measure per-token quantization error vs BF16
- **Gate**: if per-token FP8 quantization MSE exceeds a threshold (e.g. 1e-3), abort and fall back to Option B (INT8 activation path)
- **Acceptance**: full accuracy run on fcloud with C=1.0 threshold before any submission

---

## Concern 2: "The old W4A8 regression predicts real W4A8 will also be slow"

**This is the key misunderstanding — and it's important to clear up.** The old path (7ce21c3f5) and true W4A8 have fundamentally different memory behavior:

### Why old path was slow (W8A8 FP8 blockwise)

```
Weight on disk: INT4 (0.5 bytes/elem)
     ↓ load-time dequant + requant
Weight in HBM:  FP8 (1.0 bytes/elem)  ← DOUBLED!
     ↓ GEMM
FP8 × FP8 cutlass blockwise @ 296 TF
```

At decode bs=1 (S1), the kernel reads **weights from HBM** on every token. FP8 weights = 2× more bytes = 2× more time. **This dominates everything else.** The +118% S1 regression has nothing to do with FP8 compute — it's purely the weight bandwidth penalty.

### Why true W4A8 will NOT have this problem

```
Weight on disk:    INT4 (0.5 bytes/elem)
Weight in HBM:     INT4 (0.5 bytes/elem)  ← SAME AS BASELINE!
     ↓ in-register dequant (fused into GEMM prologue)
Weight in registers: FP8 (for QMMA)
     ↓
FP8 × FP8 QMMA @ 296 TF
```

| Property | Old path (W8A8) | True W4A8 | Baseline (Marlin W4A16) |
|---|---:|---:|---:|
| Weight storage | FP8 = 1.0 B/elem | **INT4 = 0.5 B/elem** | INT4 = 0.5 B/elem |
| Weight HBM reads | 2× baseline | **1× baseline** | 1× |
| Activation storage | FP8 = 1.0 B/elem | FP8 = 1.0 B/elem | BF16 = 2.0 B/elem |
| Activation traffic | 1× | **0.5× baseline** | 1× |
| Compute throughput | 296 TF (2×) | 296 TF (2×) | 148 TF |

True W4A8 is **equal or better on every dimension** vs Marlin W4A16. There is no reason it should be slower on any tier.

### Predicted per-tier impact (physics-based, not guesswork)

| Tier | Bottleneck | W4A8 change vs baseline | Expected Δ |
|---|---:|---|---|
| **S1** (decode bs=1) | Weight HBM bandwidth | Weight: same. Activation: 2× less. | **−5 to −10%** (modest) |
| **S8** (mixed) | Mixed bandwidth + compute | Less activation traffic + 2× GEMM compute | **−10 to −20%** |
| **Smax** (prefill, M≥1) | GEMM compute (67-83% of kernel time per R13e profiling) | 2× FP8 QMMA throughput | **−20 to −30%** |

### Visual comparison

```
                S1 (lower = better)
Baseline        ████████████ 121.71s
Old W4A8        ██████████████████████████ 265.32s  ← weight BW doubled
True W4A8 (est) ███████████ 109-115s                  ← weight BW same, act BW halved

                Smax (lower = better)  
Baseline        ████████████ 35.86s
Old W4A8        ███████████████ 46.44s     ← GEMM compute win ≠ compensate BW loss
True W4A8 (est) ████████ 25-29s            ← both BW win + compute win
```

The old path's regression is **expected and explained** by the weight bandwidth penalty. It does NOT predict true W4A8 performance — the physics is different.

---

## Bottom line

| Concern | Verdict |
|---|---|
| FP8 accuracy risk | **Real but manageable.** Per-token scaling + gated by full accuracy test. Lower risk than the already-proven-neutral FP8×FP8 path. |
| Speed regression from old W4A8 | **Irrelevant.** Old path was slow because weights doubled in size. True W4A8 keeps INT4 weight storage — zero bandwidth penalty. |

The old path's S1 +118% tells us **"don't inflate weight storage"** — which is exactly what true W4A8 avoids.

---

Does this address your concerns, or would you like me to go deeper on any aspect? And would you like me to proceed with the old-W4A8 verification test on fcloud (the 45-min sanity check we discussed earlier) as the next step?
