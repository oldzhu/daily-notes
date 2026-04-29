Ran terminal command:  rm -rf /tmp/sglang_v1 /tmp/sglang_v18 && mkdir -p /tmp/sglang_v1 /tmp/sglang_v18 && cd /home/oldzhu/sglang/benchmark/soar/demo_sala && tar -xzf minicpm_sala_submit_v1_007.tar.gz -C /tmp/sglang_v1 && tar -xzf minicpm_sala_submit_v18.tar.gz -C /tmp/sglang_v18 && cd /tmp/sglang_v1 && mv sglang sglang_root 2>/dev/null; true && ls /tmp/sglang_v1 /tmp/sglang_v18 | head

Ran terminal command:  diff -rq /tmp/sglang_v1/sglang_root /tmp/sglang_v18/sglang 2>&1 | head -60

Ran terminal command:  for f in layers/attention/minicpm_backend.py layers/attention/minicpm_sparse_kernels.py layers/attention/minicpm_sparse_utils.py models/minicpm.py server_args.py; do
  A=/tmp/sglang_v1/sglang_root/python/sglang/srt/$f
  B=/tmp/sglang_v18/sglang/python/sglang/srt/$f
  echo "=== $f ==="
  diff "$A" "$B" | wc -l
  echo "--v1 size / v18 size--"
  wc -l "$A" "$B" | head -2
done

Ran terminal command:  for f in layers/attention/minicpm_backend.py layers/attention/minicpm_sparse_kernels.py layers/attention/minicpm_sparse_utils.py models/minicpm.py server_args.py; do
  echo "############# $f #############"
  diff /tmp/sglang_v1/sglang_root/python/sglang/srt/$f /tmp/sglang_v18/sglang/python/sglang/srt/$f
done
Completed with input:  for f in layers/attention/minicpm_backend.py layers/attention/minicpm_sparse_kernels.py layers/attention/minicpm_sparse_utils.py models/minicpm.py server_args.py; do
  echo "############# $f #############"
  diff /tmp/sglang_v1/sglang_root/python/sglang/srt/$f /tmp/sglang_v18/sglang/python/sglang/srt/$f
done

Now I have complete attribution. Let me document findings and save to memory.

## What changed between v1_007 (C=1.0) and v18 (C=0)

| Area | File | Change | Effect |
|---|---|---|---|
| **Server args** | `prepare_env.sh` | **+`--enable-torch-compile --torch-compile-max-bs 8`**, removed `--skip-server-warmup` | Speed ↓locally, accuracy drift (known: ±1.8pt from v18-A 78.71 vs v18-B 80.51) |
| **Model numerics** | `models/minicpm.py` | **Removed FP32 cast around fused_qk_norm_rope** (2 places: `q,k = q.float(),k.float()` ... `q,k = q.to(orig_dtype),k.to(orig_dtype)`) | QK-norm now runs in BF16 under FP8 KV → **potential accuracy regression** |
| Sparse kernels | `minicpm_backend.py` | Adds `--sparse-topk-scale` (default 1 = same) | None (v18 uses `--force-dense-minicpm`, sparse path not hit) |
| Sparse kernels | `minicpm_sparse_kernels.py` | int32→int64 overflow fix + k_scale for FP8 KV | None (dense mode) |
| Sparse utils | `minicpm_sparse_utils.py` | cu_seqlens_q GQA OOB fix (CHANGE_0075) | None (dense mode) |
| Preset refactor | `preprocess_model.py` | Adds `sparse_qkv_w8_o_proj_w8` option | None (we use `sparse_qkv_w8`, same code path) |
| Cosmetic | `models/minicpm.py` | `*= residual_scale` instead of `= h * scale` | None |
| CLI | `server_args.py` | `--sparse-topk-scale` arg (default 1) | None |

**Conclusion**: Only **TWO** behaviorally-relevant v18-only changes could cause the regression:
1. **`--enable-torch-compile --torch-compile-max-bs 8`** — already known to shift accuracy (v18-A/B/C drift pattern correlates with it; also reduces Smax breathing room because compiled graphs may not match observed batch sizes under load)
2. **Removed FP32 cast around fused_qk_norm_rope in models/minicpm.py** — QK-norm arithmetic now runs in BF16 instead of promoted FP32; with FP8 KV cache this loses a bit of headroom

Everything else in v18 is either sparse-mode-only (and we force dense) or cosmetic.

Let me save findings + propose the A/B test, then commit the copilot-instructions update for this discovery. 

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/v1_007_vs_v18_regression_diff.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#1-1), lines 1 to 15

Replacing 1 lines with 12 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/TEST_RESULTS_TRACKING.md && git commit -m "docs: attribute v1_007→v18 regression to 2 specific code deltas

Local diff of both packages shows in dense+FP8-KV mode (our submission path)
only these v18 deltas are behaviorally active:

1. --enable-torch-compile --torch-compile-max-bs 8 added in prepare_env.sh
2. FP32 cast around fused_qk_norm_rope removed in models/minicpm.py (2 sites)

All other v18 changes (preprocess_model preset refactor, int32->int64 sparse
kernel overflow fix, k_scale FP8 descaling, cu_seqlens GQA fix, sparse-topk-scale
arg) affect sparse-mode code paths only, which v18 does not use
(--force-dense-minicpm).

Primary regression suspect: removed FP32 qk-norm cast under FP8 KV cache loses
~8 mantissa bits in Q@K on long-context (70K+ tokens dominate private eval).
Secondary suspect: torch.compile known accuracy drift + graph-mismatch penalty
at high concurrency (explains Smax blowup).

Attribution saved to /memories/session/v1_007_vs_v18_regression_diff.md." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Ran terminal command:  rm -rf /tmp/sglang_v1 /tmp/sglang_v18 /tmp/pkg_v1_007 /tmp/pkg_v18

## Key finding: regression attribution is now surgical

Local diff of `minicpm_sala_submit_v1_007.tar.gz` vs `minicpm_sala_submit_v18.tar.gz` identified exactly **two** behaviorally-active changes in the dense + FP8 KV submission path (since both use `--force-dense-minicpm`, all sparse-mode code changes are dead code):

1. **`prepare_env.sh`**: v18 added `--enable-torch-compile --torch-compile-max-bs 8` and removed `--skip-server-warmup`
2. **`models/minicpm.py`**: v18 removed the FP32 cast wrapper around `fused_qk_norm_rope` (2 call sites). v1_007 did `q,k = q.float(),k.float(); fused_qk_norm_rope(...); q,k = q.to(orig_dtype),...`; v18 runs QK-norm directly in BF16.

All other diffs (`preprocess_model.py` preset refactor, `minicpm_sparse_kernels.py` int32→int64 + k_scale FP8 descaling, `minicpm_sparse_utils.py` cu_seqlens GQA fix, `minicpm_backend.py` + `server_args.py` `--sparse-topk-scale`) only execute on the sparse-attention path, which v18 submission never enters.

## Primary suspect: the removed FP32 qk-norm cast

This is the more concerning of the two. With FP8 KV cache already in play, the dequantized K vectors have roughly 7-bit mantissa resolution. The QK-norm computation (`RMSNorm` on per-head vectors) amplifies small errors, and the Q@K dot-product then sums over 70K+ tokens for QA/CWE/FWE/NIAH tasks (which dominate the accuracy set). Without the FP32 promotion, rounding errors accumulate and can flip close-margin logits. **This is exactly the kind of bug that shows as "drifting accuracy" on long-context eval while barely touching short-context mcq.**

## Secondary suspect: torch.compile

Already has a documented ±1-2pt accuracy drift (v18-A/B/C = 78.71/80.51/76.64). Also a plausible contributor to the Smax blowup because compiled graphs are specialized for observed batch sizes; when the official eval runs at variable concurrency with contention, graph-spec mismatches trigger recompile or fallback.

## Proposed A/B plan (on fcloud, one session)

Four variants to pin down which delta matters:

| Variant | torch.compile | FP32 qk-norm cast | Expected |
|---|---|---|---|
| A | v1_007 as-is | OFF | ON | baseline, acc ≥ 78%, S1 ~120s local |
| B | v18 as-is | ON | OFF | current, acc ~77-78% local |
| C | v18 w/o torch.compile | OFF | OFF | isolates FP32-cast impact |
| D | v18 with FP32 cast restored | ON | ON | isolates torch.compile impact |

Most likely outcome: **D wins** → we re-add the FP32 cast and keep torch.compile. Best of both worlds. New "best baseline" is then v18 + FP32 cast restored. That baseline gets Iteration A-0 on top.

## Implementation (3 ~1-line edits)

Once you're ready to start fcloud, the D variant is trivial:

**minicpm.py** (our working tree, mixed_minicpm_cudagraph branch) — restore 2 call sites:
```python
orig_dtype = q.dtype
q, k = q.float(), k.float()
fused_qk_norm_rope(q, k, ...)
q, k = q.to(orig_dtype), k.to(orig_dtype)
```

That's it. No rebuild needed (it's pure Python), just resync to fcloud `/root/submission_sim/sglang/python/`.

Variant C = D minus the torch.compile line in prepare_env.sh.

## Awaiting your decision

1. **Approve the A/B plan as described?** If yes, I'll draft a bilingual proposal doc and exact edit sequence.
2. **Or skip A/B and go straight to D** (re-add FP32 cast + keep torch.compile) as hypothesis-driven fix, then submit v19 with Iteration A-0 mcq fix bundled? This is faster (1 submission cycle instead of 4) but less diagnostic.
3. **Or A+B only** (2 variants, just swap `submission_sim` dirs — no code edits) to first confirm v1_007 actually does beat v18 locally? Lowest effort, least risk.

My recommendation: **option 3 first**, then **option 2** (go straight to D for v19). That way we spend one fcloud session confirming the hypothesis (A vs B), then one submission cycle validating the fix officially.

No fcloud activity started — instance still shut down. Awaiting your call.  



