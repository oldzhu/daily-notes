I’m implementing the approved sparse-layer QKV W8 feature as a separate preset, keeping it fused-group-safe and layer-aware rather than guessing individual layer IDs.

Searched for files matching `sglang` · `docs/soar_2026_changes/CHANGE_0066*`, no matches

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#60-60), lines 60 to 140

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#390-390), lines 390 to 460

I’ve confirmed the current preset code only supports `o_proj_w8`. I’m extending it with a config-driven `sparse_qkv_w8` preset that derives sparse-layer IDs from `mixer_types`, while keeping `o_proj_w8` and `off` available for comparison.

Generating patch (439 lines) in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py), [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0066_gptq_sparse_qkv_w8.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0066_gptq_sparse_qkv_w8.zh.md)


```

```

```

```

```

```

```

```
The sparse-layer QKV W8 preset is in place. I’m validating syntax and basic diagnostics now, then I’ll summarize the exact env knobs you can use for full sparse versus narrowed sparse subsets.

Ran terminal command:  set -e && bash -n benchmark/soar/demo_sala/prepare_env.sh && python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

Checked [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh), [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0066_gptq_sparse_qkv_w8.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0066_gptq_sparse_qkv_w8.zh.md), no problems found

Implemented the approved sparse-layer QKV W8 feature. The preprocess path in preprocess_model.py now supports a `sparse_qkv_w8` preset that derives sparse/full-attention layer IDs from `config.mixer_types` and promotes `q_proj`, `k_proj`, and `v_proj` together to W8 only on those layers. prepare_env.sh now defaults to that preset and exposes optional controls for narrowing the sparse subset. I also added the required bilingual docs in CHANGE_0066_gptq_sparse_qkv_w8.en.md and CHANGE_0066_gptq_sparse_qkv_w8.zh.md.

The main env knobs are:
- Default experiment: `SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8`
- W8 settings: `SOAR_GPTQ_SPARSE_QKV_BITS=8`, `SOAR_GPTQ_SPARSE_QKV_GROUP_SIZE=128`
- Baseline fallback: `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
- Optional narrower subset: `SOAR_GPTQ_SPARSE_LAYER_IDS=...`

Shell and Python syntax validation passed. I did not run preprocess or evaluation locally.

Natural next steps:
1. Test the full sparse-layer preset first.
2. If accuracy improves but speed regresses too much, rerun with a narrowed `SOAR_GPTQ_SPARSE_LAYER_IDS` subset instead of abandoning the direction immediately.
3. If you want, I can commit and push this once you’re ready to run it on fcloud.

Made changes.
