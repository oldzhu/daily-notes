## Comparison of (A) vs (B)

### (A) Memory cleanup + reduced calibration

**Pros**
- Minimal code change (~20 lines): a few `del` + `torch.cuda.empty_cache()` calls plus two config knobs.
- Uses modelopt's `export_hf_checkpoint` exactly as designed → output layout is guaranteed correct.
- Zero risk to accuracy from export-side custom code.
- Fast to implement and to try; if it works we're done. If it doesn't, we still learn the actual peak memory needed (we can log `torch.cuda.max_memory_allocated()`), which tells us whether (B) is even necessary.
- Easy to roll back.

**Cons**
- May not be enough. We don't know the true peak memory of stage 2 export. If the dominant cost is *per-Linear* (modelopt holds fp16 + fp4 + fp8-scale copies of one big Linear simultaneously), no amount of calibration trimming helps — peak is set by the largest layer (e.g. `gate_up_proj` 4096×32768 fp16 = 256 MB, plus copies × per-tensor scratch buffers can balloon).
- Reducing calibration samples from 90 → 16 may slightly hurt scale quality (small accuracy risk, ~0.1–0.5 pp). FOS scale selection itself is unaffected.

### (B) Layer-streaming export

**Pros**
- Solves the OOM root cause structurally: peak memory becomes one decoder layer instead of the whole model.
- Robust to future model size growth.
- Doesn't depend on guessing how much memory we need to free.

**Cons**
- ~150–300 lines of code that drives modelopt's internals (`real_quantize_weights`, the safetensors writer, the `hf_quant_config.json` emitter) per-layer. Modelopt does not expose a clean per-layer API.
- High chance of subtle bugs: layer ordering in safetensors, sharding metadata, missing global tensors (embeddings, lm_head, norms), and crucially **the resmooth step is global** — fusing QKV scales across the whole layer can't be done in a strict streaming pass without first collecting all amaxes.
- Higher risk of producing a checkpoint that loads but is silently miscalibrated.
- Much harder to debug on fcloud (long iteration cycle).

### Verdict: **(A) is better — try it first.**

Reasoning:
1. The information value of (A) is high even if it fails: it tells us actual peak memory, which is the input we need to design (B) correctly. Going to (B) blind is premature optimization.
2. (A) is hours; (B) is days. Competition time pressure favors the cheap probe.
3. (A) keeps modelopt as the single source of truth for export layout, which is the only known way today to produce a sglang-compatible NVFP4 checkpoint. (B) reimplements that, with all the risk that entails.
4. If (A) fails by a small margin (e.g. peak 90 GB on 84 GB GPU), we can do *targeted* CPU-offload of only the largest 1–2 layers — a much smaller change than full streaming.

**Plan**: revert Path C's no-op resmooth patch and CPU-move; in the export block add (a) explicit cleanup of pre-export tensors, (b) `torch.cuda.empty_cache()`, (c) lower calibration samples to 16 and `max_calib_seq_len` to 1024, (d) log `max_memory_allocated` before and after export. Push. Ask you to resume fcloud and re-run.

Want me to implement (A)?
