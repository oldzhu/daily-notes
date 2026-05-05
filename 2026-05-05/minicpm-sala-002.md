Searched for regex `exclude|sparse_qkv|include_modules|dynamic_rules|w8|bits=8|skip` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#434-434), lines 434 to 540

Searched for regex `_resolve_sparse_layer_ids` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#416-416), lines 416 to 434

Searched for regex `exclude_modules\s*=|SOAR_GPTQ_EXCLUDE|GPTQ_EXCLUDE` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#855-855), lines 855 to 880

Good catch — let me clarify the actual config and confirm impact:

**Current baseline (`SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8`)**

Looking at preprocess_model.py and preprocess_model.py:

- **Std-attn (sparse) `q/k/v/qkv_proj`** → **INT8 GPTQ** (not BF16). They are still quantized, just at 8 bits to recover accuracy.
- **Std-attn `o_proj`** → INT4 GPTQ (default 4-bit).
- **MLP `gate/up/down_proj`** → INT4 GPTQ.
- **Lightning Q/K/V/O** → INT4 GPTQ.
- **Truly BF16 (excluded entirely)**: only `self_attn.o_gate` and `self_attn.z_proj` (lightning output-gate / z projection — not my tagged set).

**Impact on my W4A8 hook — none, by design.**

The bit-width guard in gptq.py `_soar_maybe_setup_w4a8_fp8`:

```python
if c.weight_type.size_bits != 4:
    return  # INT8 layers stay on Marlin
```

automatically skips INT8 layers. So with `sparse_qkv_w8`:

| Layer | Bits | What happens |
|---|---|---|
| std-attn qkv_proj | INT8 | stays on Marlin INT8 (BF16 tensor cores) — unchanged |
| std-attn o_proj | INT4 | **switches to FP8 GEMM (QMMA)** ✅ |
| MLP gate_up/down | INT4 | **switches to FP8 GEMM (QMMA)** ✅ |
| lightning q/k/v/o | INT4 | stays on Marlin INT4 (untagged, per amendment #3) |
| o_gate / z_proj | BF16 | unchanged (no GPTQMarlinLinearMethod) |

**Verification once on fcloud**: count `[SOAR W4A8] enabled` lines in the server log — expected ~7 lines per std-attn layer (3 = MLP × 1, plus o_proj × 1 = 2 actually... wait, per *layer* it's `gate_up_proj` (1 fused), `down_proj` (1), `o_proj` (1) = 3 lines. There are 8 std-attn layers, so **24 std-attn entries**, plus 24 lightning layers contribute MLP only → wait, MLP is per *decoder layer* regardless of mixer type, so 32 layers × 3 (gate_up, down) ... actually let me recount: every layer has MLP (32 × 2 linears = 64), plus 8 std-attn layers contribute o_proj (qkv excluded by INT8 guard) = 8 → **72 lines total**.

**Net speed gain ≈ slightly less than originally projected**, because std-attn QKV (which has the fattest matmuls) stays on BF16 Marlin. If we want to also accelerate those, a **follow-up iteration** can add an INT8→FP8 conversion path (sgl-kernel still has `fp8_blockwise_scaled_mm` accept FP8 weights regardless of original bit-width — same dequant→requantize trick, just from W8 instead of W4). But that's out of scope for this iteration; want to keep one-feature-per-iteration as per workflow rules.

Safe to start fcloud whenever you're ready. 


