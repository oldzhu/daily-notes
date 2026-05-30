**Yes, your understanding is correct.** With `--force-dense-minicpm`:

- `model_config.has_sparse_attention = False`
- Pool selected = plain `HybridReqToTokenPool` (no `req_to_sparse_k1_token`, no `req_to_sparse_k2_token`)
- `cache_finished_req`'s `isinstance(..., MiniCPMReqToTokenPool/MiniCPMHybridReqToTokenPool)` check → `False` → K1/K2 branch never entered
- The whole crash class is structurally impossible

---

**Why it was removed:** When `SOAR_BACKEND_VARIANT=flashinfer` was adopted (Round 13f-1, v20), the reasoning was:

> Stock flashinfer doesn't invoke the `minicpm_backend` at all — `--force-dense-minicpm` was originally needed to prevent the `minicpm_flashinfer` backend from running sparse attention kernels. With stock flashinfer, that flag is irrelevant for attention compute.

So it was dropped as "harmless to remove". But the side effect nobody considered at the time: **it still affects pool type**. Dropping it → `MiniCPMHybridReqToTokenPool` gets instantiated → K1/K2 slots allocated during every prefill, even though stock flashinfer never uses them → silent wasted memory + latent MEDUSA crash.

The toggle `SOAR_BACKEND_KEEP_FORCE_DENSE=1` already exists in `prepare_env.sh` to restore it.

---

**Proposed test:**

Set `SOAR_BACKEND_KEEP_FORCE_DENSE=1` on fcloud before sourcing `prepare_env.sh`. This restores `--force-dense-minicpm` (and `--dense-as-sparse`, which stock flashinfer ignores). MEDUSA stays active (`SOAR_SPEC_MEDUSA=1`). Expected benefits:
- Crash class eliminated at root (not patched — just structurally gone)
- Slightly less GPU memory wasted (no K1/K2 tables)

Commands to run (after you approve starting fcloud):
```bash
# On fcloud:
export SOAR_BACKEND_KEEP_FORCE_DENSE=1
source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT" "${SGLANG_SERVER_ARGS[@]}"
```

Then accuracy + S1/S8/Smax tests.

**Do you approve? I'll start the fcloud instance and run the full test suite.**
