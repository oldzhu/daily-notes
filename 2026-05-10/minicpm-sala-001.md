Created 3 todos

Searched for files matching `**/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`, 1 match

Searched for regex `attn_impl|attention_backend|flash_attention_2|force_dense|sparse_config|mixer_types|FORCE_DENSE` (`**/benchmark/soar/demo_sala/**`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#880-880), lines 880 to 920

Got the answer. Now check `_prepare_gptq_load_source` to see if it modifies the model config (force-dense, sparse stripping) for calibration.     

Searched for regex `_prepare_gptq_load_source|force_dense|sparse_config|mixer_types|MiniCPMSALAForCausalLM` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 7 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#777-777), lines 777 to 880

Searched for regex `_sanitize_model_config_for_gptq|FORCE_DENSE` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#738-738), lines 738 to 775

Searched for regex `SOAR_GPTQ_FORCE_DENSE|force_dense` (`**/benchmark/soar/demo_sala/**`), 16 results

Searched for regex `sparse_config|attn_implementation|MiniCPMAttention|forward|register` (`**/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py)

Searched for regex `save_pretrained|save_quantized|config_post|sparse_config|json.dump|to_json|copy.*config` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 20 results

Before running anything on fcloud, here's my analysis and proposal.

## Q: Does GPTQ calibration use sglang's `minicpm_flashinfer` backend on the mixed (sparse+lightning) layers?

**No — calibration is pure HuggingFace transformers, not sglang.** Specifically:

1. preprocess_model.py calls `GPTQModel.load(src, ..., attn_implementation="flash_attention_2")`. This loads the model via HF transformers using the model repo's `trust_remote_code` modeling files (`modeling_minicpm_sala.py`).
2. Activation collection runs forward passes through that HF model on calibration prompts. **sglang is not involved at all** during calibration — it only enters at server-launch time after the quantized weights are saved.
3. Per-layer behavior during HF calibration:
   - **Lightning layers**: model's native PyTorch `MiniCPMLightningMixer` / chunked-GLA path.
   - **`minicpm4` (sparse) layers**: HF's `attn_implementation="flash_attention_2"` → standard **dense softmax** attention (NOT sglang's top-k sparse attention). The HF modeling code doesn't have the top-k sparse kernel; that lives only in sglang's `MiniCPMSparseBackend`.
4. preprocess_model.py only touches `rope_scaling` / `rope_type`. It does NOT modify `sparse_config`, `mixer_types`, or anything sparse-related. `SOAR_GPTQ_FORCE_DENSE` is exported but **never read** by preprocess_model.py — only consumed at sglang server launch time as `--force-dense-minicpm`.
5. Net: GPTQ collects q/k/v/o + MLP statistics on layers 0/9/16/17/22/29/30/31 under **dense flash-attention activations**, not under sparse-top-k activations. The `sparse_qkv_w8` preset just bumps those layers' weight bit-width to 8, but the calibration distributions themselves are dense-softmax.

This is a **subtle source of the GPTQ-sparse-FP8 accuracy regression**: scales/zeros for q/k/v on the sparse layers are fitted to dense-softmax statistics, but inference applies them under top-k sparse attention where the descale rounding compounds differently in the top-k score path. The mismatch is quantization-flow inherent, not just an FP8 KV artifact.

## Round 13e proposal — three deliverables

### (1) Non-GPTQ sparse + FP8 KV smoke test (PRIMARY — needs your approval)

**Goal**: separate "GPTQ × sparse interaction is broken" from "current sglang sparse path is globally broken/slow".

**Config** (use unquantized `/root/models/openbmb/MiniCPM-SALA`, BF16 weights):
- `--attention-backend minicpm_flashinfer`
- `--kv-cache-dtype fp8_e5m2`
- NO `--quantization gptq_marlin`, NO `--force-dense-minicpm`, NO `--enable-torch-compile` (sparse + compile is broken on HEAD)
- `--max-running-requests 8 --mem-fraction-static 0.78` (BF16 weights ≈ 16 GB vs GPTQ ≈ 6 GB, so KV pool gets squeezed; also reduces tail-latency risk)
- `--disable-radix-cache --enable-mixed-chunk`
- `--dense-as-sparse` (matches Test 9 historical config)
- Accuracy concurrency = **8** (not 32) — sparse path is slow per-request; concurrency=32 saturates and creates 50min tails. C=8 reduces tail risk and is closer to S8 official tier anyway.

**Decision matrix** based on outcome:
| Result | Diagnosis | Next action |
|---|---|---|
| Accuracy ≥ 79% AND completes in < 1h | sparse path itself is fine; **regression is GPTQ-specific** | Bisect GPTQ load path / Marlin × sparse interaction |
| Accuracy < 78% but completes | sparse + FP8 KV is fundamentally noisy on bf16 too (descale issue is universal, not GPTQ-specific) | Sparse + FP8 KV is a dead end; pivot to sparse + bf16 KV (Test 9 territory) |
| Times out / runs > 1.5 h | sparse path is **globally slow** on current HEAD (regression in sglang) | Bisect sparse-attention kernel commits between `9d3ecd168` and HEAD |

**Estimated time**: 30–60 min for accuracy, no speed test in this round.

### (2) Regression-bisect plan (deferred to after Test 1 result)
Once we know which bucket the regression lives in (GPTQ vs global), narrow the search:
```bash
git log 9d3ecd168..a4e394085 --oneline -- \
  python/sglang/srt/layers/attention/minicpm_backend.py \
  python/sglang/srt/layers/attention/minicpm_sparse_utils.py \
  python/sglang/srt/mem_cache/memory_pool.py \
  python/sglang/srt/model_executor/cuda_graph_runner.py \
  python/sglang/srt/models/minicpm.py
```
Then revert candidate commits as **python-only patches** (no wheel rebuild needed for these files) and retest in a fast loop.

### (3) Improving-baseline criterion
We can claim a viable sparse + FP8 KV baseline if and only if:
- Normalized accuracy ≥ 97% (C ≠ 0)
- AND S1 + S8 + Smax all complete within 2× of dense baseline (otherwise official scoring will hurt more than the accuracy win)

If after fixing the regression we still fail (3), sparse stays parked.

---

**Approval needed**: shall I proceed with **(1)** — the non-GPTQ sparse + FP8 KV accuracy test on `/root/models/openbmb/MiniCPM-SALA` at concurrency=8? You confirmed fcloud is started. 

Completed: *Answer GPTQ calibration backend question* (1/3)


