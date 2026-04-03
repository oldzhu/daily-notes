I’m implementing the approved combined preset by composing the existing sparse-QKV and `o_proj` W8 override paths, then I’ll add the required CHANGE_0068 EN/ZH docs and run syntax checks.

Searched for files matching `sglang` · `docs/soar_2026_changes/CHANGE_0068*`, no matches

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#420-420), lines 420 to 520

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#102-102), lines 102 to 155

I’m patching the preset parser to support a true combined mode rather than comma-separated names, and I’ll make the new combined preset the default for the next test so your fcloud run uses it without extra env changes.

Generating patch (337 lines) in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py), [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0068_gptq_sparse_qkv_w8_o_proj_w8.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0068_gptq_sparse_qkv_w8_o_proj_w8.zh.md)


```

```

```

```

```

```

```

```
The combined preset is implemented and set as the default for the next run. I’m validating syntax now so you can move straight to fcloud testing if it passes.

Ran terminal command:  set -e && bash -n benchmark/soar/demo_sala/prepare_env.sh && python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

Checked [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh), [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0068_gptq_sparse_qkv_w8_o_proj_w8.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0068_gptq_sparse_qkv_w8_o_proj_w8.zh.md), no problems found

Implemented the combined preset you asked for.

The preprocess logic in preprocess_model.py now supports:
- `off`
- `o_proj_w8`
- `sparse_qkv_w8`
- `sparse_qkv_w8_o_proj_w8`

and prepare_env.sh now defaults to:
- `SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8_o_proj_w8`

So your next fcloud run will use sparse-layer QKV W8 plus global `o_proj` W8 unless you override it. I also added the required bilingual docs in CHANGE_0068_gptq_sparse_qkv_w8_o_proj_w8.en.md and CHANGE_0068_gptq_sparse_qkv_w8_o_proj_w8.zh.md.

Syntax validation passed. I did not run preprocess or evaluation locally.

For quick comparison on fcloud:
- Combined preset: default as-is
- Sparse-only: `export SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8`
- O-proj-only: `export SOAR_GPTQ_MIXED_PRECISION_PRESET=o_proj_w8`
- Baseline: `export SOAR_GPTQ_MIXED_PRECISION_PRESET=off`

If you want, I can commit and push this next.

