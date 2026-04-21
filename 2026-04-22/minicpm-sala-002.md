## G3 — Dense GPTQ Recalibration Explained

You're right that GPTQ calibration was already done during 4-bit quantization. The point of G3 is: **what data/mode was used during that calibration?**

GPTQ works by passing calibration data through the model and measuring activation statistics to minimize quantization error. The key question is:
- If the GPTQ model (`sparse_qkv_w8`) was calibrated with **sparse attention** active (the default mode), the activation distributions it optimized for differ from what happens in **dense mode** (`--force-dense-minicpm`)
- Dense attention produces different intermediate activations (all tokens attend to all, vs sparse topk selection), so the "optimal" quantization grid shifts slightly

In practice, for W4A16 (weights-only quantization), this effect is small since only weights are quantized, not activations. The gain would be marginal (2-5% at best). **It's correctly low priority — I'd skip it too.**

---

## Plan for Your 5 Chosen Optimizations

Priority order (low-risk first, building toward high-impact):

| Order | ID | Optimization | Expected Gain | Effort |
|-------|----|-------------|---------------|--------|
| 1 | **M1** | Residual scale folding into weights | 1-3% | Low, zero risk |
| 2 | **K4** | Fused RMSNorm + residual_scale CUDA kernel | 1-2% | Medium |
| 3 | **A1** | SimpleGLA state contiguity guarantee | 5-8% decode | Medium |
| 4 | **A3** | Fuse state I/O into FLA kernel | 10-15% decode | High |
| 5 | — | Custom Triton decode kernel for SimpleGLA | 10-20% | Very high |

Note: #4 and #5 overlap significantly — a custom Triton decode kernel would inherently fuse state I/O. We may merge them into one effort.

**Cumulative potential**: ~25-40% faster if all succeed, which could bring our score from 56.63 toward the 79.55 target.

Shall I start with the **M1 (residual scale folding)** proposal? It's the safest starting point — pure math equivalence, load-time only, no runtime risk.
