Let me first review what we already have documented and identify gaps.  

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/sparse_attention_testing_status.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/sparse_attention_testing_status.md)

Let me also check the `sparse_topk_scale` parameter type — the user asked for 1.25 and 1.5 but it might be integer-only: 

Searched for text `sparse_topk_scale` (`**/python/sglang/srt/server_args.py`), 1 result

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#530-530), lines 530 to 545

It's `int` — 1.25 and 1.5 would truncate to 1. And we already know topk_scale=2 breaks everything. Let me also check how it's used: 

Searched for text `topk_scale` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 5 results

Now I have the full picture. Let me build the comprehensive matrix:

---

## Comprehensive Configuration Matrix & Gap Analysis

Here's every combination mapped against what we've tested:

| # | Model | KV dtype | Attention | topk_scale | Test # | Accuracy | Status |
|---|-------|----------|-----------|------------|--------|----------|--------|
| 1 | Non-quant | FP8 | Sparse | 1 | Test 1 | 53.18% | **Tainted** (had CHANGE_0070 bug) — needs re-test |
| 2 | Non-quant | bf16 | Sparse | 1 | Test 3/4 | **83.02%** | **Done** (clean baseline) |
| 3 | Non-quant | FP8 | Dense | 1 | — | — | **MISSING** |
| 4 | Non-quant | bf16 | Dense | 1 | — | — | **MISSING** |
| 5 | GPTQ | FP8 | Dense | 1 | Test 7 | 78.84% | Done (old fcloud, old GPTQ model) |
| 6 | GPTQ | bf16 | Dense | 1 | — | — | **MISSING** |
| 7 | GPTQ | FP8 | Sparse | 1 | Test 8b | **76.07%** | Done (new fcloud) |
| 8 | GPTQ | bf16 | Sparse | 1 | Test 9 | **79.67%** | **Done** (best config, C=0.8) |
| 9 | GPTQ | FP8 | Sparse | 2 | Test 8 | OOM | Done (crashes) |
| 10 | GPTQ | bf16 | Sparse | 2 | Test 10 | 0.20% | Done (garbage) |

**About topk_scale=1.25 and 1.5**: The `sparse_topk_scale` parameter is declared as `int`, so 1.25→1 and 1.5→1 (truncated). The only integer values between 1 and 2 are... none. And scale=2 is catastrophically broken (both OOM with FP8 and garbage output with bf16). So **topk_scale variants 1.25/1.5 are not testable** without first changing the type to `float`.

### What needs testing (4 configs):

| Priority | Config | Why |
|----------|--------|-----|
| **Must** | Non-quant + FP8 KV + sparse (re-test without 0070 bug) | Test 1 was tainted |
| **Must** | Non-quant + FP8 + dense | Complete the non-quant matrix |
| **Must** | Non-quant + bf16 + dense | Complete the non-quant matrix |
| **Must** | GPTQ + bf16 + dense | Complete the GPTQ matrix |

### Top 3 options to improve from current best (Test 9: 79.67%, C=0.8):

1. **GPTQ + bf16 KV + sparse (current best, Test 9)** — already at 99.58% normalized. Focus on speed optimization (S1/S8/Smax tuning, server scheduling params) to maximize `0.8 × speed_score`. Only needs 0.42pp more accuracy for C=1.0.

2. **GPTQ + bf16 KV + dense** — untested but Test 7 (GPTQ+FP8+dense) got 78.84%. With bf16 KV (which consistently helps), this could surpass 80% and potentially reach C=1.0. Dense also avoids all sparse complexity.

3. **Non-quant + bf16 KV + sparse (Test 3: 83.02%)** — highest accuracy by far but no GPTQ speed benefit. If speed is less critical than accuracy coefficient, this could yield C=1.0 easily.

Shall I proceed with running these 4 missing tests? I'll need you to **start the fcloud instance** first. I can change prepare_env.sh for each config, restart server, run accuracy + speed, then move to the next.
