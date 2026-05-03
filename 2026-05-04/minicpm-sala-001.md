let us discuss and research more on the way to improve and optimize  speed in minicpm mixed layers architecture(8 sparse-attention layer + 24 lighting attention layer)
1. 4-bit quantization(already dong with gptq w4a8?).
2. 4 bit kv-cache (in plan and only apply for 8 layers?)
3. for 24 lightning layers(it is maba like, right, so it ), what new or existing ways can improve(not like sparse attention or full attention it is n square complex, mamb like attention is n linear complex,am I right? from formula or algorihtms involved in the two different type of attention point of view, what diffeent ways can be used to improve? for memory intensive(decode) or compute intensive(prefill) )

=======

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#280-280), lines 280 to 410

Searched for regex `simple_gla|SimpleGLA|chunk_simple_gla|fused_recurrent` (`**/python/sglang/srt/models/minicpm.py`), 12 results

Searched for regex `chunk_simple_gla|fused_recurrent|class SimpleGLA|def forward|state_shape|prefill|decode` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 20 results

Searched for regex `class SimpleGLAAttnBackend|def forward|chunk_simple_gla\(|fused_recurrent_simple_gla\(` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1632-1632), lines 1632 to 1700

Searched for regex `bits|sym|group_size|desc_act|act_order|W8|w8|sparse_qkv` (`**/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`), no results

Searched for regex `bits|sym=|group_size|desc_act|sparse_qkv` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 20 results

I have enough technical context. Let me give a structured analysis.

# Speed-optimization research: MiniCPM-SALA mixed architecture (8 std-attn + 24 lightning)

## 0. Architecture facts to anchor the discussion

| Layer type | Count | Compute model | KV/state | Complexity (seq len `N`) |
|---|---|---|---|---|
| **`minicpm4` (standard attention)** | **8** | softmax(QKᵀ)V with paged KV cache, RoPE, GQA | Paged KV (FP8 e5m2 today, 84 GB GDDR7 budget) | Prefill **O(N²·d)** compute; Decode **O(N·d)** memory-bound (KV bandwidth) |
| **`lightning` (SimpleGLA / linear attention)** | **24** | `chunk_simple_gla` for prefill, `fused_recurrent_simple_gla` for decode; recurrent state `S = (g·S + kᵀv)`, `o = qS` | **Fixed-size** state `(num_kv_heads, head_dim, head_dim)` per layer per request; **NO seq-len-dependent KV** | Prefill **O(N·d²)** ≈ linear in N; Decode **O(d²)** **per token, independent of N** |

This is the single most important fact: **as input length grows, the standard 8 layers dominate cost more and more** (their KV grows linearly; lightning state is constant). So at the long-context regime that matters in official scoring, optimizing the 8 layers gives the biggest leverage. But the 24 lightning layers still dominate **at small context / decode-with-short-history** because they're 75% of the layer count and their per-token cost is constant `d²`.

---

## 1. Your point #1 — "4-bit quantization (already W4A8?)"

**Not quite.** What we ship today:

- GPTQ **W4A16** for most linear layers (weight 4-bit, activation BF16).
- GPTQ **W8A16** for QKV of the 8 std-attn ("sparse") layers (`SOAR_GPTQ_SPARSE_QKV_BITS=8`, `group_size=128`).
- KV cache **FP8 e5m2** (separate from weight quant).
- Activations are **BF16 throughout the GEMMs** (Marlin W4A16 / W8A16 kernels).

So we have **W4/W8 weight only**, not W4A8 and not W4A4. Real W4A8 (INT8 activations + INT4 weights via QQQ-style or Marlin-W4A8) and W4A4 (NVFP4 W4A4) are separate optimization vectors:

| Variant | Weight | Activation | SM120 expected gain | Status here |
|---|---|---|---|---|
| W4A16 (current) | INT4 | BF16 | 1× | ✅ submitted |
| W8A16 (sparse QKV) | INT8 | BF16 | 0.6× (slower than W4A16) | ✅ kept for accuracy |
| **W4A8 INT8 act** | INT4 | INT8 | ~1.4× (FP8 GEMM at 296 TFLOPS) | ❌ not tried — needs activation calibration + Marlin W4A8 kernel path |
| **W4A8 FP8 act** (mxfp8 / QMMA) | INT4 | FP8 | ~1.5-1.8× (uses QMMA on SM120) | ❌ not tried — most promising on Blackwell |
| **NVFP4 W4A4** | FP4 | FP4 | ~2.5× theoretical (593 TFLOPS) | ❌ Test 21 catastrophic acc ~12% — destroys reasoning |

**Optimization opportunity (new)**: **W4A8 with FP8 activations** is the obvious unexplored direction. SM120 has hardware QMMA for mxfp8 — using it for activations while keeping weights at GPTQ INT4 would roughly double GEMM throughput vs current W4A16 with little expected accuracy loss (FP8 e4m3 activations preserve ~99% of BF16 accuracy on most LLMs). This is much safer than NVFP4 W4A4 because activations stay in FP8 precision (which we already proved survives in our KV cache).

---

## 2. Your point #2 — "4-bit KV cache (only for 8 layers?)"

**Correct on both counts.** Lightning layers don't have a token-indexed KV cache — they have a fixed `d×d` recurrent state that doesn't benefit from "per-token quantization". So 4-bit KV applies **only to the 8 standard layers**.

Why 4-bit KV is interesting on SM120:
- KV bandwidth is the dominant decode bottleneck (RTX PRO 6000 = 1398 GB/s). Halving KV size from FP8→FP4 halves the bandwidth pressure.
- SGLang already has `MHATokenToKVPoolFP4` and `--kv-cache-dtype fp4_e2m1` upstream. M2.0 ablation proposal already addresses this.

**Why "mixed" matters specifically** (champion W5 finding restated for our model):
- First/last few layers are accuracy-critical (attention sinks, output decisions).
- Middle layers are surprisingly tolerant to FP4 KV.
- Our 8 std-attn layers are NOT contiguous — they're interspersed in `mixer_types`. So we need to identify per-position sensitivity by layer-index-within-the-8, not by absolute layer id.

**Open opt vector beyond M2.0**: even within the 8 layers, KV cache **per-channel/per-head FP4 with FP8 outliers**. Not all attention heads are equally sensitive. A "tiered" KV (FP4 for low-entropy heads, FP8 for high-entropy) could hit ~95% of FP4 bandwidth gain at 99% of FP8 accuracy. This is more sophisticated than the M2.0 ablation but lives in the same code path.

---

## 3. Your point #3 — Linear-attention layer optimization (the deep one)

Yes, lightning ≈ Mamba/GLA family — **linear** in N, not quadratic. Algorithmic structure:

```
S_t = g_t · S_{t-1} + k_tᵀ v_t      # O(d²) state update
o_t = q_t · S_t                       # O(d²) read
```

Here `S ∈ R^(d×d)` is fixed-size. **There is no seq-len axis to compress.**

### 3a. What this means for the two regimes

| Regime | Bottleneck | What helps lightning |
|---|---|---|
| **Prefill (compute-bound)** | The chunked algorithm `chunk_simple_gla` — Triton kernel doing block-sparse matmuls; cost ≈ `O(N · d²)` GEMM-equivalent | (i) larger chunk size for better tensor-core utilization (we already have `SGLANG_FLA_CHUNK_SIZE` per CHANGE_0080), (ii) FP8/FP4 GEMM in the chunk kernel itself, (iii) fusing q/k norm + RoPE + GLA prelude (we have `fused_qk_norm_rope`) |
| **Decode (memory-bound on weights, NOT on state)** | `fused_recurrent_simple_gla` — for each token: load `S` (`d²` words), do 2 small matmuls, store `S` back. Total per layer = O(d²) reads/writes. Across 24 layers + per-request batching, **state load/store traffic** + **weight load** dominates | (i) keep `S` in registers/SMEM across the whole layer (already via `fused_recurrent`), (ii) **quantize the `S` state itself** (FP16→FP8 → halves state BW), (iii) batch-merged kernel that does Q/K/V proj + recurrent update + O proj in one launch (kernel-fusion) |

### 3b. Algorithmic optimization vectors specific to linear attention

These are the levers that DON'T exist for standard attention:

1. **State quantization (FP16 state → FP8 state)** — biggest, untouched lever.
   - Each layer keeps `S ∈ R^(num_kv_heads × d × d)` in BF16/FP32. With `d=128` (typical), that's ~64KB/layer/req. 24 layers × bs=24 ≈ 36 MB of state — fits in L2 (112 MB), but **state load/store dominates the recurrent kernel inner loop**.
   - FP8 state would halve that traffic. Accuracy impact is unknown but plausibly small because the state is heavily averaged (geometric decay `g`).
   - Implementation: modify `fused_recurrent_simple_gla` to accept a quantization scale per head. Probably a 1-2 day project on the kernel side.

2. **State re-materialization across requests / streaming-ssm tricks** — for very long-context decode, reset state cadence vs. recompute trade-off. Our context is bounded by 128k so probably moot here.

3. **Chunk-size auto-tuning** — `chunk_simple_gla` performance is highly sensitive to chunk size relative to head_dim and SM count (96 SMs on SM120). CHANGE_0080 made chunk size a knob; we should sweep it on the **official-style long-context speed dataset** (not our local short-context one). Likely a 5-15% prefill win at no accuracy cost.

4. **Chunk vs. recurrent crossover threshold tuning** — `_select_mode` in `SimpleGLAAttnBackend` chooses chunk mode for prefill, fused_recurrent for decode. There's a sweet spot for **chunked extend with small new tokens** (e.g., chunked-prefill where the new chunk is 64-256 tokens). Currently SGLang picks chunk mode any time `is_extend`. For very small extend chunks the overhead of chunk-launch may exceed recurrent cost — a runtime threshold could pick the cheaper path.

5. **Kernel-fusion: QKV-proj + GLA + O-proj in one Triton kernel** — for small batch decode, the launch + register-fill overhead dominates. A single fused kernel `compute_lightning_layer(x_in, W_qkv, W_o, S_inout) -> x_out` would cut 3 kernel launches to 1. This is a significant rewrite (~1-2 weeks) but worth it given lightning is 24/32 of the layers.

6. **Matrix-form parallel decode (speculative-decoding's free friend)** — when decoding `g` tokens via speculative draft, lightning layers can absorb all `g` tokens with a single `chunk_simple_gla` call instead of `g` recurrent calls. Linear attention is the **ideal speculative-decoding target** — verification cost is amortized for free. This lifts speculative decoding from "hard win" to "easy win" specifically for our architecture.

7. **Output gate / RMSNorm fusion** — we already have `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE`. Verify it's actually picked up on the v18 path; it's another small constant win.

### 3c. What does NOT help lightning (don't waste time)

- ❌ Sparse attention / topk attention — only meaningful for quadratic softmax attention.
- ❌ KV cache compression — there's no token-indexed KV.
- ❌ Flash-attention v2/v3 — those are for softmax attention.
- ❌ Sliding window — irrelevant for linear-attention since seq-len is already collapsed in `S`.

---

## 4. Concrete priority ranking for our codebase

Sorted by `(expected_gain × probability) / effort`, all on top of the v18 baseline:

| # | Optimization | Layers affected | Expected gain | Effort | Risk | Why now |
|---|---|---|---|---|---|---|
| **1** | **W4A8 FP8 activations (mxfp8 QMMA)** | All linear layers | **+30-50% GEMM throughput** | 1-2 weeks (new Marlin kernel path) | Medium | SM120 has the hardware; current 296 TFLOPS FP8 unused |
| **2** | **M2.0 → M2.1 mixed FP8/FP4 KV** | 8 std-attn | +10-15% decode TPS, less KV memory | Low (already proposed; 50 LOC) | Low (per-layer ablation finds safe layers) | Already drafted |
| **3** | **Lightning state FP8 quantization** | 24 lightning | +5-10% decode TPS (whole model) | Medium (kernel mod) | Medium-low | Untouched lever; biggest win on linear path |
| **4** | **Fused lightning layer (QKV + GLA + O) Triton kernel** | 24 lightning | +5-15% latency at small bs | High (~2 weeks) | Medium | Reduces launch overhead at low bs |
| **5** | **Chunk-size sweep + crossover threshold** | 24 lightning | +3-7% prefill | Low (env var sweep) | Low | Cheap exploration |
| **6** | **Speculative decoding (n=2-3) leveraging linear-attn cheapness** | All | +20-40% decode TPS at low bs | Medium (draft model + integration) | Medium | Long-tail mcq issue overlaps with this |
| **7** | **Per-head/per-channel tiered FP4 KV** | 8 std-attn | +5-10% over uniform FP4 | Medium | Medium | Only after M2.0 lands |

**Recommendation**: stay on M2.0 (#2) since it's already drafted, then go to **#1 (W4A8 FP8) and #3 (lightning state FP8)** as the next two parallel work streams. Those two combined could realistically bring us to mid-tier of the leaderboard. #6 (speculative decoding) is the only path to top-5 without major architectural changes.

Want me to draft a formal proposal for any of #1, #3, or #4?   


