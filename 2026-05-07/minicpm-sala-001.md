### Baseline (v18) flow — W4A16 BF16

Yes, the current baseline is **W4A16 with BF16 activations**, running the GPTQ-Marlin path:

1. **Load time**: GPTQ checkpoint stays as **packed INT4 weights in HBM** (no dequant at load). Group scales kept as **fp16/bf16**. Marlin repacks the INT4 layout for its kernel-friendly format. **No weight inflation** — weight bytes in HBM = INT4 (the win).
2. **Forward path** (per Marlin GEMM call):
   - Activations are **BF16** (no activation quantization).
   - Marlin K-loop: load packed INT4 from HBM → in-register dequant `INT4 → BF16` using fp16/bf16 group scale → MMA `BF16 × BF16 → FP32 accumulator` on tensor cores at **148 TF** → epilogue cast to BF16.
3. **KV cache**: FP8 e5m2 storage (set via `--kv-cache-dtype fp8_e5m2`); K/V dequantized to BF16 inside FlashAttention before the attention matmul.

So in three lines:
- **Weights**: INT4 in HBM → in-kernel dequant to BF16 → MMA in BF16
- **Activations**: BF16 throughout (no quant)
- **KV**: FP8 in HBM → dequant to BF16 in attention

Compared to the W8A8-FP8 test we ran:

| Aspect | v18 W4A16 BF16 (baseline) | W4A8 #1 mislabel (= W8A8 FP8) |
|---|---|---|
| Weight HBM dtype | INT4 packed | **FP8 e4m3** (inflated 2×) |
| Weight bytes per param | 0.5 B | 1.0 B |
| Dequant in K-loop | INT4 → BF16 | none (already FP8) |
| MMA | BF16 × BF16, 148 TF | FP8 × FP8, 281 TF peak |
| Activation dtype | BF16 | FP8 e4m3 (per-token quant on each forward) |
| Decode (bandwidth-bound) | Fast (W4 storage) | **Slower** (W8 storage = 2× weight reads) |
| Compute (compute-bound regime) | 148 TF ceiling | 281 TF ceiling, but rarely reached at SOAR shapes |

That's why the W8A8 FP8 test regressed: 2× compute peak doesn't help much in decode (which is weight-bandwidth-bound), and we paid 2× weight bandwidth.

The **ideal W4A8 FP8** (Proposal A in PROPOSAL_W4A8_REAL_001 / parked) would keep INT4 in HBM (no inflation) AND get the FP8 MMA — but requires the 3–4 week kernel rewrite.

---

Now patching the docs to record this clarification. 

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_002.en.md#1-1), lines 1 to 50

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md#1-1), lines 1 to 50

Replacing 16 lines with 94 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md)


```

```

```

```

```

```

```

```
Now create the Round 8 chat log and commit.  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-clarifications-baseline_20260428_0930.en.md)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-clarifications-baseline_20260428_0930.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.* docs/soar_2026_changes/PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.* docs/soar_2026_changes/chat/CHAT_w4a8-clarifications-baseline_20260428_0930.* && git commit -m "Clarify W8A8 FP8 vs INT8 history, Phase 0 vs Proposal B distinction, baseline (W4A16 BF16) flow" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -8

Pushed `c4192653b`.

## Recap

**Baseline v18 (W4A16 BF16)** flow:
1. **Load**: GPTQ INT4 stays packed in HBM, fp16/bf16 group scales kept. **No weight inflation.**
2. **Forward**: Activations BF16 (no quant). Marlin K-loop dequants `INT4 → BF16` in registers, then `BF16 × BF16 → FP32` MMA at 148 TF, BF16 epilogue.
3. **KV**: FP8 e5m2 in HBM, dequantized to BF16 inside FlashAttention.

**Doc patches applied** (commit `c4192653b`):
- ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md § 7 / zh § 7 — added 3-way comparison table (baseline / W4A8 #1 / real W4A8) + Phase 0 vs Proposal B.
- PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md § 1.1 / zh § 1.1 — added "How this differs from Phase 0" with side-by-side table.
- New chat log: CHAT_w4a8-clarifications-baseline_20260428_0930.en.md / zh.

Awaiting your decision: NVFP4 KV proposal (P1 survey first) and/or 1-day W4-FP8 spike?        



