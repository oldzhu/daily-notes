Good question. Honest answer: **theory says longer-is-better, but our actual evidence points the other way for this specific model**. Let me lay out both sides.

## Theory — why longer SHOULD be better

NVFP4 uses **per-tensor activation amax** captured during calibration. The `input_quantizer` divides activations by `amax/6.0` to fit them into FP4's [-6, 6] range. So calibration must observe the *maximum* activation magnitude that will appear at inference.

For a long-context model evaluated up to 128K tokens:
- At seqlen=4096, calibration only sees activation magnitudes from short prefills.
- Attention activations grow with sequence length (RoPE positions, long-range KV interactions, RMSNorm/LayerNorm stats per token).
- If inference activations exceed calibration amax → values clip to ±6 in FP4 representation → catastrophic accuracy loss on long-context tasks (niah, fwe).
- **Theoretical prediction**: bumping calib from 4096 → 16384 should HELP niah/fwe, not hurt.

## Empirical — what we actually observed

| Run | calib seqlen | scheduling | niah | fwe | qa | ori |
|-----|---:|---|---:|---:|---:|---:|
| Iter 1 run 1 | 4096 | Tier1 | 93.33 | 92.22 | 56.67 | 75.98 |
| Iter 1 run 2 | 4096 | Tier1 | 83.33 | 96.67 | 50.00 | 70.27 |
| Iter 2 run 1 | 16384 | conservative | 73.33 | 70.00 | 50.00 | 60.73 |
| Iter 2 run 2 | 16384 | conservative | 70.00 | 82.22 | 36.67 | 63.31 |

niah dropped 93→73 and fwe dropped 92→70 — **opposite of theory**. Three plausible reasons:

1. **Calibration stratification changed alongside seqlen** — iter 1 used SOAR_GPTQ_CALIBRATION defaults (32 sequential, no task filter). Iter 2 used 90 stratified qa,mcq,cwe. So the iter-1 calib set probably *had* niah/fwe samples (long-context, big amax) and iter 2 *excluded* them. The amax for long-range layers may actually be *lower* in iter-2 calib than iter-1, even though seqlen is longer. → **calibration content matters more than seqlen**.

2. **Outlier-driven amax**: NVFP4 per-tensor scaling is dominated by the single largest activation seen. Adding more long samples can *raise* amax → wider FP4 dynamic range → coarser resolution for typical activations → all tasks degrade. Long calib is a **lossy averaging** of dynamic ranges.

3. **Iter 2 confounded scheduling AND calibration** — we cannot attribute the regression to seqlen alone. That's exactly what iter 3 (current plan, same calib + iter-1 scheduling) is meant to disentangle.

## My recommendation

**It must be empirical.** There's no a-priori "longer better" or "shorter better" — it depends on:
- Whether amax saturates within seqlen=4096 for this model (probably yes for QKV, maybe no for o_proj/down_proj which see longer contexts).
- Whether the calibration sample mix already contains long-context examples (more important than seqlen).
- Whether we use per-tensor (one amax) or per-channel (more capacity to absorb outliers) scaling — modelopt NVFP4_DEFAULT_CFG uses per-tensor for activations.

Concrete decision tree once iter 3 results are in:
- **If iter 3 (16384 calib + iter-1 sched) ≥ 75%** → seqlen is fine, scheduling owned the iter-2 regression. Stop here, FOS works.
- **If iter 3 < 70%** → calibration content/seqlen is the lever. Then iter 4 (re-quant at **4096** + iter-1 sched) tests "shorter is better" hypothesis.
- **If iter 3 in [70, 75)** → mid-band; both knobs contribute. Run iter 4 anyway.

If iter 4 wins big over iter 3 → "shorter calib + iter-1 sched" is the recipe → ship.
If iter 3 ≈ iter 4 → calib seqlen doesn't matter much; the iter-2 regression was scheduling all along.

So your "abort if <70% then go to seqlen=4096" rule is exactly the right experimental design. Approve to start step 1?
