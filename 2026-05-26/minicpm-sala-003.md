Your intuition about representational capacity is **correct**. Achieved accuracy is a different number, and that's where it broke.

## Representational capacity: yes, NVFP4-FOS > GPTQ-INT4 at the same 4 bits/weight

| | GPTQ INT4 g=128 | NVFP4 block=16 | NVFP4 + FOS |
|---|---|---|---|
| Scales per 4096-K row | 32 (FP16) | 256 (FP8) | 256 (FP8) |
| Scale entries vs INT4 | 1× | **8×** | 8× |
| Per-block tunable | scale only | scale only | scale + M ∈ {4,6} |
| Levels per element | 16 uniform | 16 non-uniform E2M1 | 16 non-uniform E2M1 |

(Note: 8×, not 4× — block 16 vs group 128 → 128/16 = 8× more scales.)

So **on paper** NVFP4-FOS can fit the weight tensor strictly better than GPTQ-INT4. If you took the same BF16 weight and quantized it independently with both schemes and measured `‖W_q − W‖`, NVFP4-FOS would win.

## Why our test still lost ~10 pp

Three multiplicative losses that representational capacity doesn't see:

**1. W4A4 vs W4A16 — biggest factor.** This is *not* about weights. NVFP4_DEFAULT_CFG quantizes the **activation** to FP4 every forward. INT4 (Marlin) keeps activations in BF16. Per-layer activation rounding error is squared and accumulated over 80 layers. Even if your weights are perfect, you've added a noise floor to every Linear's input. This alone is multi-pp.

**2. Calibration: static rounding vs Hessian-aware rounding.** Higher capacity only translates to lower achieved error if you *use* the capacity well. Modelopt's NVFP4 does:
```
  for each block:  scale = max(|w|) / 6  (or /4 with FOS),  q = round(w / scale)
```
GPTQ does:
```
  H = X^T X  (calibration Hessian)
  for each column k:
      q_k = round(w_k / scale)            # local rounding
      e_k = w_k - q_k * scale             # residual
      W[:, k+1:] -= H_inv * e_k * H[k, k+1:]   # propagate residual into still-unquantized columns
```
GPTQ uses `H` to *steer* the rounding choices toward minimizing **layer output error on real activations**, not weight L2. NVFP4-FOS picks M=4 vs M=6 per block by **weight L2 only** — no `X`, no `H`. So GPTQ at lower capacity still beats NVFP4-FOS at higher capacity.

**3. sparse_qkv_w8.** GPTQ baseline upgrades 24 sensitive linears from W4 g=128 to W8 g=128 — those layers go from 16 levels to 256 levels. NVFP4-FOS in iter-5/iter-7 was uniform W4 everywhere. So on the 24 layers that matter most, GPTQ's effective capacity is **256-level / 128-group**, NVFP4's is **16-level / 16-block**. INT4-W8-mixed beats FP4-uniform-W4 on those layers regardless of block size.

## Why the champion's recipe is appealing

"GPTQ + NVFP4 levels + Marlin W4A16 decode" is exactly what neutralizes losses 1 and 2 while keeping the capacity advantage:

- **Loss 1 gone**: Marlin W4A16 means BF16 activations.
- **Loss 2 gone**: still using GPTQ's Hessian-aware OBQ, just snapping to FP4 levels at the rounding step.
- **Loss 3 still applies**: would need a sparse_qkv_w8 analog on top, but at least the W4 layers get FP4 levels.

So your reasoning was right; what you were missing is that the path that *uses* NVFP4-FOS (modelopt's full pipeline) bundles it with W4A4 + static calibration + uniform W4. The capacity advantage from 16-vs-128 block size and FOS gets entirely consumed by the other three losses, and then some.

A clean way to think about it:

- **Capacity** says how well you *could* fit the weights.
- **Calibration algorithm** says how well you *actually* fit them.
- **Activation precision** says how much noise the inference path adds on top.
- **Mixed precision** says whether sensitive layers get more capacity.

NVFP4-FOS wins on capacity, loses on the other three. INT4-GPTQ-sparse_qkv_w8 wins on the other three, loses on capacity. Net: the other three weigh more than capacity.

So you're not seeing a contradiction — you've found the actual reason NVFP4 vendors keep saying "use it weight-only with a W4A16 kernel." Which loops back to the champion's exact recipe.
