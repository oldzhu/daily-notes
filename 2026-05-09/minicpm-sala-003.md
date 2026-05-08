Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "cat /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/config.json" 2>&1 | head -200

Got the config. Confirmed your expectation:

## Model architecture: 32 layers = 8 sparse + 24 lightning

**`mixer_types` breakdown** (32 layers total):
- `minicpm4` (sparse) at indices: **[0, 9, 16, 17, 22, 29, 30, 31]** = **8 layers**
- `lightning-attn` at the other **24 layers**

**`quantization_config.dynamic`** bumps `q/k/v/qkv_proj` **only on those 8 sparse layers** to `bits=8` (group_size=128). All other weights stay `bits=4`. That's exactly the `sparse_qkv_w8` naming — sparse layers get W8 because they drive the top-k scoring path which is precision-sensitive.

**Lightning layers don't have a paged KV cache** — they use a GLA hidden state (`SimpleGLACacheParams`). So `--kv-cache-dtype fp8_e5m2` (or any KV dtype) **only affects those 8 sparse layers**. This is why FP8-KV-breaks-sparse is so localized: the descale error compounds in the very 8 layers that score top-k.

This sharpens our picture significantly:

| Path | What `--kv-cache-dtype fp8_e5m2` actually quantizes |
|---|---|
| **Dense + force-dense-minicpm** (current baseline) | KV of all 8 minicpm4 layers, used as full-KV softmax (no top-k). Lightning layers unaffected. |
| **Sparse (no force-dense)** | KV of all 8 minicpm4 layers, used both for **top-k scoring** (descale → low-precision dot product → bad indices) AND for the actual selected-block attention. Lightning unaffected. |

Why dense tolerates FP8 KV but sparse doesn't is now mechanical: dense uses K/V once (softmax over all positions, errors average out); sparse uses K twice — once for scoring (where rounding picks wrong blocks → cliff effect) and once for attention.

## Proposal — sparse + FP8 KV retest plan

**Goal**: confirm/refute the Test 8b accuracy regression on current code, AND collect S1/S8/Smax for the sparse path (a real data gap).

**Config** (one fcloud round):
- Edit `prepare_env.sh` `SGLANG_SERVER_ARGS`:
  - **Remove** `--force-dense-minicpm`
  - **Keep** `--kv-cache-dtype fp8_e5m2`
  - **Restore** `--attention-backend minicpm_flashinfer` (default for sparse)
  - Keep `--enable-torch-compile --torch-compile-max-bs 8 --enable-mixed-chunk` (matches v18 baseline minus the dense flag)
- Total args change: 1 removed, 1 added.

**Runs**:
1. `accuracy` (~50 min) → ori_acc, normalized, C, per-task scores
2. `speed --variant all` (~5 min) → S1, S8, Smax

**Decision tree**:
- **Acc ≥ 78%** AND speed ≥ baseline → champion-recipe hypothesis confirmed; pivot toward sparse + FP8 (or sparse + mixed KV next).
- **Acc < 77%** (Test 8b reproduces): sparse + FP8 dead. **But** speed might still be informative — if Smax is much faster, it motivates spending effort on sparse + bf16 KV (or sparse + mixed KV: bf16 on K, FP8 on V).
- **Acc 77-78%** (borderline C=0.92): same as above but more nuanced.

**Risk**: low. Reversible. ~1 hour fcloud.

**Approval needed**: should I (a) edit `prepare_env.sh`, sync, restart server, run accuracy, then speed; or (b) just edit `prepare_env.sh` locally and let you review before pushing?

I'd recommend (a) — full pipeline is the cheapest way to get the answer. Let me know. 


