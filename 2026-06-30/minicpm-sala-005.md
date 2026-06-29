Good instinct — let me review the history before we commit to either lever. Let me check the W4A8 iteration docs and the earlier results. 

Searched for files matching `**/docs/soar_2026_changes/CHANGE_W4A8_001*.md`, 8 matches

Searched for regex `ldmatrix|dequant.*SMEM|broadcast|double.*memory|memory footprint|smem.*dequant|shared memory.*dequant` (`**/docs/soar_2026_changes/**/*.md`), 20 results

Great question — and your instinct is partly right. Let me separate two different "double memory" concepts, because the history shows they're not the same thing.

## The historical regression you're remembering

That was **W4A8#1** (2026-04-27, the documented `REGRESSION`: S1 +118%, S8 +56%, Smax +30% vs Marlin). But its root cause was **HBM weight storage doubling**, not SMEM:

> "implementation upcast INT4 weights to **FP8 storage at load time**, doubling weight memory footprint vs Marlin INT4 baseline. At decode bs=1 the kernel is memory-bandwidth-bound on weights → FP8 weight = 2× slower than INT4 weight."

That's a **global-memory (HBM)** problem: 4-bit → 8-bit weights in DRAM → 2× the bytes the decode loop must stream. The two levers I listed are **SMEM-only** — the weights stay INT4 in HBM — so they do **not** reproduce that regression. Good news on that front.

## But the key finding: dequant-once-to-SMEM was already tested — it's v1

Here's the part that validates your skepticism. "Dequant-once-to-SMEM-then-broadcast" is **exactly what v1 already does**:

```cuda
// v1 Phase 1: all 128 threads cooperatively dequant the whole tile ONCE
for (int i = tid; i < kTileN * kTileK; i += blockDim.x) { ... W_fp8[...] = val; }
// then all 4 warps read the shared FP8 buffer
```

So we already have the head-to-head data:

| Kernel | Dequant strategy | TFLOPS (M=16384) |
|---|---|---|
| **v1** | **dequant once → FP8 SMEM, broadcast to all warps** | **39.2** |
| v2 | register dequant (4× redundant), no SMEM | 37.3 |
| v3 | register dequant (4× redundant) + cp.async pipeline | 44.5 |

**v1 ≈ v2.** The SMEM-broadcast strategy (no redundancy) and the register strategy (4× redundancy) land in the same place. That means the **4× redundant dequant is NOT the bottleneck** — so removing it via SMEM broadcast buys ~nothing, while it *would* cost SMEM.

## And yes — it would raise SMEM and likely hurt

You're right about the memory pressure, just in SMEM not HBM. To combine "dequant-once-to-SMEM" with v3's pipeline you'd need **both** buffers per stage:

- v3 today: 3 stages × 12.6 KB (INT4 + acts + scales) = **~37 KB**
- adding an FP8 dequant buffer: +8 KB × (buffering) → **~45-61 KB**

SM120 caps dynamic SMEM ~100 KB/SM. At 37 KB you can *maybe* fit 2 blocks/SM; pushing to ~60 KB drops you to **1 block/SM — halving occupancy**. Given v1≈v2 already proved the broadcast gives no throughput benefit, paying occupancy for it is a **net loss**. So: **don't pursue dequant-to-SMEM.** The history backs you up.

## ldmatrix is different — memory-neutral, but low ceiling

`ldmatrix` does **not** double anything — it reads from the *same* activation SMEM buffer, just in the tensor-core-native swizzled layout instead of scalar `memcpy`. So no memory concern there. But it only optimizes the **A-fragment load**, which is a small slice of the cost, so the ceiling is modest.

## Honest reassessment of the gap to Marlin

The evidence (v1≈v2, pipeline gave the only real win) points away from dequant/load micro-opts. The remaining ~0.55× gap is more likely:

1. **Occupancy** — v3's 37 KB SMEM already limits blocks/SM; Marlin runs leaner.
2. **Per-element FP8 conversion cost** — `__nv_fp8_e4m3(float)` in the dequant has multi-cycle latency × every weight element, every K-tile.
3. **The pre-existing ~0.90-cosine correctness gap** — which still needs root-causing before *any* speed number on this kernel is trustworthy.

My recommendation, given all this: **don't spend effort on dequant-to-SMEM or ldmatrix.** Either (a) root-cause the correctness gap first (highest value — an incorrect kernel can't ship regardless of speed), or (b) accept v3 as the documented ceiling of the manual approach and return to the v25 cutlass path / catalog. 

I'll correct the "next levers" note in the tracking doc since the v1≈v2 evidence actually argues against the dequant-to-SMEM suggestion I wrote there. Want me to do that, and which direction — (a) correctness root-cause, or (b) park it?
