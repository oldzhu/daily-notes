You are right — switching paths instead of fixing the root cause is a smell. Let me explain the three paths honestly, including which switch was justified vs which was avoidance.

## What modelopt's NVFP4 export actually does

`export_hf_checkpoint(model)` runs three stages internally:
1. **Resmooth/fuse** — `requantize_resmooth_fused_llm_layers` runs a *dummy forward pass* (`model(input_ids=[[1,2]])`) to recompute amax statistics across fused QKV/MLP layers.
2. **Compress weights** — for each Linear, calls `tensor_quantizer.real_quantize_weights` which packs fp16 weight → fp4 (uint8) and produces fp8 per-block scales.
3. **Save** — writes safetensors + `hf_quant_config.json`.

The standard sglang-loadable layout is: `weight (N, K/2) uint8`, `weight_scale (N, K/16) fp8_e4m3`. Stage 2 must produce that exact shape.

## The three paths

### Path A — `mtq.compress(model)` then GPU export
- **Why I tried it**: modelopt 0.43 added `mtq.compress` as a "pre-compress before export" helper that I assumed would just do stage 2 early.
- **What broke**: produces `weight_scale` with shape `(N, K/32)` — half the expected K dim. `mtq.compress` is for a *different* compressed serialization format that sglang's loader doesn't support.
- **Was the switch justified?** Yes. `mtq.compress` is the wrong API; using it was my mistake. The fix is **don't call `mtq.compress`**, just call `export_hf_checkpoint`. That's actually what Path B does.

### Path B — drop `mtq.compress`, but `model.to("cpu")` to avoid OOM
- **Why**: without `mtq.compress`, `export_hf_checkpoint` keeps both fp16 and fp4 copies of weights in GPU memory during stage 2. On 84 GB this OOMs for MiniCPM-SALA. Moving the model to CPU before export sidesteps the OOM.
- **What broke**: Stage 1 (resmooth) does a dummy forward, which on CPU calls MiniCPM-SALA's `fused_recurrent_simple_gla` — a Triton kernel that requires CUDA tensors.
- **Was the switch to Path C justified?** **No, this was avoidance.** The proper fixes for Path B are, in order:
  1. Free GPU memory **before** export: delete optimizer state, calibration activations, KV cache, then run export on GPU. Likely fits.
  2. Reduce calibration sample count and clear `m._input_quantizer._amax_buffer` style scratch tensors.
  3. Selectively move only large Linear weights to CPU, but keep the model graph on CUDA so the dummy forward works.
  4. Stream layer-by-layer: move one decoder layer to CPU for compression, write its tensors, move it back / drop it.

### Path C — keep CPU-move, monkey-patch resmooth to a no-op
- **Why**: tried to skip the CUDA-only dummy forward.
- **What broke**: process exits silently with no traceback (the current blocker).
- **Was the switch justified?** **No.** Skipping resmooth is also wrong on the merits — resmoothing recomputes correct fused-layer scales; without it the exported model's QKV/MLP scales may be miscalibrated, hurting accuracy. Even if Path C "worked" we'd have an accuracy-degraded model.

## Honest summary

| Path | Real problem | Right fix |
|------|--------------|-----------|
| A | Used wrong modelopt API | Don't use `mtq.compress` — go to standard export |
| B | GPU OOM during export | Free pre-export GPU memory; keep model on CUDA |
| C | (avoidance of B) — silent exit, and skips required resmooth | Abandon C, return to B |

**Path B is the correct path. We should fix its OOM properly rather than work around it on CPU.**

## Recommended next action (revised)

Drop Path C. Resume work on Path B with one of these minimal-risk OOM fixes (in order of effort):

1. **Free obvious memory before export**: `torch.cuda.empty_cache()`, drop calibration activation buffers, drop GPTQ-time scratch tensors. Measure free GPU mem; report it.
2. **Reduce calibration `max_calib_seq_len`** from 4096 → 1024, and `calibration_samples` from 90 → 16. We don't need 90×4096 activations cached during export.
3. **Layer-streaming export**: walk decoder layers, move each to CPU one-by-one only *after* `real_quantize_weights` has been called on its Linears (needs a small custom helper around modelopt's internals, but no monkey-patch of correctness-critical code).

Want me to:
- **(A)** Revert Path C's no-op patch from `preprocess_model.py`, add the GPU-memory-cleanup steps from option 1+2 above, push, and ask you to resume fcloud for a re-run?
- **(B)** Just revert C and go straight to layer-streaming (option 3) — more code, but the most robust?
