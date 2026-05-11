Searched for regex `sparse_dense_len|dense_as_sparse` (`**/python/sglang/srt`), no results

Searched for regex `sparse_dense_len|dense_as_sparse`, 20 results

Searched for text `sparse_config` (`**/benchmark/soar/demo_sala`), no results

Good observations on both counts. Here's the full picture:

## `dense_len` — what it is and where it comes from

- It's a **per-model config field**, not a CLI arg.
- Loaded in minicpm.py: default `512`, can be overridden by either `sparse_dense_len` at top of HF `config.json` or `sparse_config.dense_len` nested inside it.
- Fallback in model_config.py: `getattr(hf_config, "sparse_dense_len", 512)`.
- Used as a **per-request seq-len threshold**: in minicpm_sparse_utils.py — `if seq_lens_cpu[i] >= dense_len or dense_as_sparse: → sparse branch`. Otherwise dense full attention inside the same backend.

## CLI tunability

- **No** dedicated `--dense-len` flag exists. Only `--dense-as-sparse` (boolean → forces threshold to 0).
- To tune the threshold without code changes, edit the model's `config.json` (`sparse_config.dense_len` or top-level `sparse_dense_len`). This happens at preprocess/quantization time.
- We could add a `--sparse-dense-len <int>` server arg if we want runtime tuning — small change in server_args.py + minicpm_backend.py L238. ~10 lines.

## Your reasoning is correct

1. With `--attention-backend flashinfer` (official default), `--dense-as-sparse` is dead code — the MiniCPM backend isn't even loaded. Remove it for cleanliness.
2. With `--attention-backend minicpm_flashinfer`, `--dense-as-sparse` is actively harmful: it forces every short request through top-k scoring + sparse FA, which costs more than the dense FA path it would otherwise take. Sparse only pays off when `seq_len ≫ kernel_size × top_k_blocks`; below that, dense is both faster and more accurate.

So **remove `--dense-as-sparse` from `prepare_env.sh`**. The default `dense_len` from the model config will then auto-route short requests to dense and long requests to sparse — which is exactly the design intent of the mixed architecture.

## Is the default value optimal?

Default `512` is conservative (most batches < 512 tokens go dense). For our workload (long-context evals up to 128K), this is probably fine because the requests we care about for sparse benefit are all ≫ 512. But it's worth a quick sweep later (e.g., 256 / 512 / 1024 / 2048) once we have a working sparse profile from option-B.

## Proposed action

1. Edit prepare_env.sh `noquant` branch: drop `--dense-as-sparse`.
2. Note this in the upcoming CHANGE_0135 (option-B profiling) proposal so the profile run reflects realistic mixed dense/sparse routing instead of forced-sparse.
3. Defer adding a `--sparse-dense-len` CLI override until we have profile data showing the threshold matters.

Want me to make the `prepare_env.sh` edit now (one-liner) and roll it into CHANGE_0135? 


