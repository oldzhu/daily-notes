Good catches — let me address each precisely.

### 1. The W8A8-on-4-bit-model test — what actually happened

Mostly correct, but with one important nuance:

- The "W4A8 #1" test (commit `7ce21c3f5`) was implemented on the **GPTQ INT4 model**, but at load time the loader **dequantized INT4 → BF16 → re-quantized to FP8 e4m3 storage**, then ran a CUTLASS **FP8×FP8 BF16-out** GEMM (W8A8 **FP8** blockwise, not INT8).
- Activation path: BF16 → FP8 e4m3 per-token quantize on every forward.
- So this was effectively **W8A8 FP8** (not W8A8 INT8). I labeled it "W4A8" originally because it loaded a W4 GPTQ checkpoint, which is what triggered the mislabel discussion two rounds ago.
- Result: accuracy stayed OK (because BF16→FP8 round-trip is mild), but speed regressed because INT4 weights got **inflated to FP8 storage in HBM** → weight memory bandwidth doubled. Decode is bandwidth-bound → regression.

**So your statement is correct in spirit (we tested an A8 quantization on the W4 model and got worse benchmarks), but the activation/weight type was actually FP8, not INT8.**

### 2. Did we test W8A8 INT8?

**No.** We never ran W8A8 INT8 end-to-end. INT8 only came up in Phase 0 as a **synthetic microbenchmark of dense INT8 IMMA at the GEMM level** (not on the model). That microbench measured 136 TF and killed the W4-INT8 path before we built anything.

### 3. Proposal B vs Phase 0 — are they the same test?

**No, they measure different things.** Let me show the difference precisely:

| Aspect | Phase 0 (already done, 2026-04-27) | Proposal B (CUTLASS W4-FP8 spike, proposed) |
|---|---|---|
| What's measured | **Dense FP8×FP8 GEMM** (both inputs already in FP8 in registers) | **W4-weight × FP8-activation GEMM** (weights start as packed INT4 in HBM, dequantized in-kernel to FP8 before MMA) |
| Tool | `torch._scaled_mm` (cuBLASLt FP8 path) | CUTLASS W4-FP8 example or hand-rolled kernel |
| What it tells us | The hardware FP8 ceiling (281 TF measured) | Whether the **W4→FP8 dequant overhead** drops us below that ceiling |
| Why it matters | Confirms FP8 cores are real on SM120 | The actual production kernel has dequant in the K-loop — that's exactly the overhead Phase 0 did NOT measure |

Phase 0's 281 TF tells us "**if** weights and activations were both already FP8 in registers, MMA can hit 281 TF". But the real W4-FP8 kernel pays a price the Phase 0 mat-mul doesn't:

- Load packed INT4 from HBM
- Mask + shift to extract 4-bit values
- Multiply by fp16 group scale
- **Encode as FP8 e4m3 bit pattern** (sign + 4-bit exp + 3-bit mantissa, with saturation)
- Then feed MMA

That dequant-encode chain runs every K-tile, in the same SMs that are also doing the MMA. If it stalls the warps, achieved throughput drops well below 281 TF.

**The unknowns Phase 0 did NOT answer:**
- Does W4→FP8 dequant fit in register/smem budget on SM120?
- Does it overlap with MMA or serialize?
- What % of the 281 TF ceiling do we keep?

Proposal B answers exactly these unknowns. The thresholds (≥ 250 / 180–249 / < 180 TF) tell us how much of the FP8 ceiling survives the dequant overhead — which decides whether the 3–4 week full kernel is worth it.

### Suggested doc fix

I should clarify both points in the existing docs (the W8A8-was-FP8-not-INT8 history, and Phase 0 vs Proposal B distinction). Want me to:

1. **Patch ANALYSIS / PROPOSAL B docs** to explicitly contrast Phase 0 (dense FP8 ceiling) vs Proposal B (W4→FP8 dequant tax measurement)?
2. Also patch `CHANGE_W4A8_001_iteration_002` mislabel section to make clearer that the impl was W8A8 **FP8** (not INT8) since today's question shows the wording is still ambiguous?

Both are quick. Or do you want to skip docs and proceed directly with one of the proposals? 


