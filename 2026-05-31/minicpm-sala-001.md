Compacted conversationGood question. Let me look at the actual code to give a precise answer.  

Searched for text `req_to_sparse_k1_token` (`**/python/sglang/**`), 20 results

Searched for regex `force.dense.minicpm|has_sparse_attention|MiniCPMHybridReqToTokenPool|MiniCPMReqToTokenPool` (`**/python/sglang/**`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/model_config.py#234-234), lines 234 to 252

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#360-360), lines 360 to 420

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/schedule_batch.py#1470-1470), lines 1470 to 2060

Now I have a complete picture. Here's the precise answer:

---

**Your premise is partially right, but there's a mismatch between two independent things:**

## 1. Attention *computation* vs. KV *pool allocation* are separate

`--attention-backend flashinfer` controls which **kernel computes attention**. It does NOT control how the KV memory pool is typed.

The pool type is selected by model_runner_kv_cache_mixin.py based solely on `model_config.has_sparse_attention`:

```python
# model_runner_kv_cache_mixin.py:369
if self.model_config.has_sparse_attention:   # ← THIS determines pool type
    self.req_to_token_pool = MiniCPMHybridReqToTokenPool(...)  # has req_to_sparse_k1_token
else:
    self.req_to_token_pool = HybridReqToTokenPool(...)         # NO req_to_sparse_k1_token
```

And `has_sparse_attention`:

```python
# model_config.py:238
return getattr(self.hf_config, "has_sparse_attention", False) if not self.force_dense_minicpm else False
```

## 2. The default config does NOT have `--force-dense-minicpm`

Looking at prepare_env.sh, with the default `SOAR_BACKEND_VARIANT=flashinfer` and `SOAR_BACKEND_KEEP_FORCE_DENSE` unset:

```bash
FORCE_DENSE_ARG=""  # ← dropped! No --force-dense-minicpm in SGLANG_SERVER_ARGS
```

So `model_config.force_dense_minicpm = False` → `has_sparse_attention = True` → `MiniCPMHybridReqToTokenPool` → `req_to_sparse_k1_token` **exists and gets populated**.

## 3. K1/K2 slots are allocated during PREFILL regardless of backend

schedule_batch.py's `prepare_for_extend()` (lines 1473–1530) computes and calls `alloc_for_extend(...)` whenever `has_sparse_attention=True`. This writes slot IDs into `req_to_sparse_k1_token` unconditionally — even when stock flashinfer handles the actual attention compute and never touches those K1/K2 slots.

## 4. Why not crash without MEDUSA?

Without MEDUSA, `prepare_for_decode()` also allocates K1/K2 slots each decode step (lines 1983–2036). So `req_to_sparse_k1_token[pool_idx, 0:k1_total]` is fully populated through the end of the request, and `cache_finished_req` frees exactly what was allocated — no phantom frees.

**With MEDUSA**, `prepare_for_decode()` hits the early return at:
```python
if not self.spec_algorithm.is_none():
    return  # ← K1/K2 alloc for decode steps is SKIPPED
```
So `kv_committed_len` grows (decode steps increment it via `scheduler_output_processor_mixin.py`) but `req_to_sparse_k1_token` is NOT updated beyond the prefill slots. When the request finishes, `k1_total = (kv_committed_len - kernel_size) // kernel_stride + 1` exceeds the prefill-allocated count → reads stale non-zero IDs from the previous occupant → phantom frees → crash.

## Summary

| Factor | Value |
|--------|-------|
| Attention compute | Stock flashinfer (no sparse kernels used) |
| KV pool type | `MiniCPMHybridReqToTokenPool` (has K1/K2, because no `--force-dense-minicpm`) |
| K1/K2 allocated? | YES — during prefill, by `schedule_batch.prepare_for_extend` |
| K1/K2 used by attention? | NO — stock flashinfer ignores them |
| Root of the bug | K1 slots allocated at prefill time, but decode steps don't add more (MEDUSA early-return), yet `kv_committed_len` grows → `cache_finished_req` reads beyond allocated range into stale data |

If `--force-dense-minicpm` were present, `has_sparse_attention=False` → plain `HybridReqToTokenPool` → no `req_to_sparse_k1_token` → `cache_finished_req` never enters the K1/K2 branch → crash impossible. 


