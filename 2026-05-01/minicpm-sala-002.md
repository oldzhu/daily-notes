Ran terminal command:  python3 scripts/fcloud/fcloud_exec.py exec "cd /root && diff -rq submission_sim_B/sglang submission_sim_C/sglang 2>&1 | head -60; echo ---SIZES---; du -sh submission_sim_B/sglang submission_sim_C/sglang; echo ---gptqmodel_minicpm_sala---; diff submission_sim_B/gptqmodel_minicpm_sala.py submission_sim_C/gptqmodel_minicpm_sala.py | head -80"

Ran terminal command:  python3 scripts/fcloud/fcloud_exec.py exec "cd /root && diff -rq --exclude=__pycache__ submission_sim_B/sglang submission_sim_C/sglang 2>&1 | head -40"

## v19 vs v18: the regression source

### Source code diffs (sglang python, excl. pycache)
| File | Type | Impact |
|------|------|--------|
| `srt/layers/attention/fla/chunk.py` | modified | FLA kernel — **directly affects MiniCPM-SALA linear-attention numerics** |
| `srt/layers/attention/fla/chunk_delta_h.py` | modified | FLA delta-rule kernel |
| `srt/layers/attention/fla/fused_recurrent.py` | modified | FLA recurrent path |
| `srt/layers/attention/hybrid_linear_attn_backend.py` | modified | Hybrid attn dispatcher (full vs linear) |
| `srt/models/minicpm.py` | modified | MiniCPM model code |
| `srt/models/minicpm3.py` | modified | MiniCPM3 model code |
| `srt/models/minicpm_eagle3.py` | **new** | EAGLE3 speculative head for MiniCPM |
| `srt/speculative/eagle_worker.py` | modified | EAGLE worker update |

### Config diffs (`prepare_env.sh`)
```
chunked-prefill-size           32768 → 65536
max-prefill-tokens             32768 → 65536
prefill-max-requests           1     → 4
max-running-requests           20    → 24
schedule-conservativeness      1.0   → 0.8
+ --enable-mixed-chunk
+ SGLANG_FLA_CHUNK_SIZE=64
```

### Preprocess diff (`preprocess_model.py`)
```python
+ if SOAR_GPTQ_FORCE_DENSE=1 and sparse_config is not None:
+     sanitized["sparse_config"] = None   # dense-calibration pass
```
(Does **not** affect this run — we reused v18's preprocessed GPTQ model.)

### Likely culprits for the -5.75pt drop (79.51 → 73.76)

1. **FLA kernel changes + `SGLANG_FLA_CHUNK_SIZE=64`** — three FLA files changed simultaneously. Linear-attention chunking is numerically sensitive; a new chunk size may break correctness on long context. (Note: cwe/fwe/niah still ~OK, but qa and mcq dropped — mcq especially is short-context so it shouldn't be FLA-sensitive unless minicpm.py changed).
2. **`minicpm.py` / `minicpm3.py` model edits** — these touch the forward path; any layer swap or masking edit would show up in all tasks.
3. **Mixed-chunk + aggressive prefill (`--enable-mixed-chunk` + 65 K chunks + 4 concurrent prefills)** — interacts with linear-attention chunk boundaries. Also probable cause of the 3 000 s timeout at sample 132.
4. **Noise-free test**: mcq is deterministic (temperature=0), and mcq avg_out jumped from v18's 12 267 → v19's 10 658 — different truncation pattern → different thinking → different answer. Consistent with a model forward change, not just scheduling.

### Isolation plan (if you want to dig deeper)
Cheapest first (all fit in prepare_env.sh — no rebuild):
- **v19-a**: v19 tree + v18's `SGLANG_SERVER_ARGS` line (drop aggressive flags + mixed-chunk). If acc recovers → flags are the cause. If not → source code is the cause.
- **v19-b**: v19 tree + v18 `SGLANG_SERVER_ARGS` + unset `SGLANG_FLA_CHUNK_SIZE`. Isolates FLA chunk-size impact.
- **v19-c**: swap v19's `minicpm.py` / `minicpm3.py` / `fla/*.py` back to v18 copies. Isolates source-code impact.

My bet: **combination of `--enable-mixed-chunk` + `FLA_CHUNK_SIZE=64` under v19's modified FLA kernels**. The model-code path itself is probably fine (cwe/fwe/niah still good).

**Recommendation**: don't spend more fcloud time — keep **v18 as baseline** and pursue Iteration A-0 on top of v18. v19's flags look like they were tuned for the new official speed set (heavy long-context, high concurrency) but break correctness on short-context mcq/qa.

Want me to shut down fcloud now? Or run one of the v19-a/b/c isolation tests first? 


