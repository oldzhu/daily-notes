I have approval for the control-flow fix. I’m patching the hook installer so the later `transformers` hooks are always installed, then I’ll run syntax and diagnostics again.

Generating patch (207 lines) in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0053_prepare_model_rope_scaling_sanitization.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0053_prepare_model_rope_scaling_sanitization.zh.md)


```

```

```

```

```

```
The installer control-flow fix is in. I’m running the usual syntax and diagnostics checks now so you can retry with a clean entrypoint.

Ran terminal command:  cd /home/oldzhu/sglang && python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

Checked [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0053_prepare_model_rope_scaling_sanitization.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0053_prepare_model_rope_scaling_sanitization.zh.md), no problems found

Applied the control-flow fix.

What changed:
- benchmark/soar/demo_sala/preprocess_model.py
  - `_install_gptqmodel_minicpm_rope_patch()` no longer returns early after installing only the GPTQ-side hook
  - GPTQ-side hooks and `transformers`-side hooks are now installed independently in the same run
  - added an installer summary log:
    - `[preprocess] Installed MiniCPM-SALA load compatibility hooks hooks=[...]`
  - kept the late version prints close to:
    - `transformers.from_pretrained(...)`
    - `transformers._from_config(...)`
- docs/soar_2026_changes/CHANGE_0053_prepare_model_rope_scaling_sanitization.en.md
- docs/soar_2026_changes/CHANGE_0053_prepare_model_rope_scaling_sanitization.zh.md
  - updated to document the installer control-flow fix and hook summary

Validation:
- `python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py`
- diagnostics on touched files: no errors

What to look for on the next run:
1. Hook installer summary:
   - `[preprocess] Installed MiniCPM-SALA load compatibility hooks hooks=[...]`
2. Late version line near the actual load boundary:
   - `[preprocess] GPTQ dependency versions pre_transformers_from_pretrained ...`
   - or `pre_transformers_from_config ...`
3. Late config snapshots:
   - `[preprocess][rope-debug] transformers_from_pretrained_entry_pre_patch ...`
   - `[preprocess][rope-debug] transformers_from_pretrained_entry_post_patch ...`

If those lines still do not appear, then we know the failing path is bypassing even the patched `transformers` entrypoints, which would be a very different conclusion. If they do appear, they should finally tell us whether `default` is still present at the actual crash boundary.

