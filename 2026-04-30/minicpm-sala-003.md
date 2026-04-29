Searched for files matching `**/docs/soar_2026_changes/PROPOSAL_iteration_A0*`, 2 matches

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_iteration_A0_mcq_runaway.en.md)

Here's both:

---

## Tomorrow's v19 plan (3-way A/B/C)

Same mechanism as today but with a third extract. I'll pre-stage on fcloud startup:

```
/root/submission_sim_A  ← v1_007   (pre-v18, verified C=1.0 officially)
/root/submission_sim_B  ← v18      (current eliminated C=0, locally 79.51%)
/root/submission_sim_C  ← v19      (never tested, Apr-20 build)
```

Swap `/root/submission_sim` symlink between runs, use the SAME preprocessed GPTQ model at `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` (v1_007/v18 already confirmed identical `prepare_model.sh` + `gptqmodel_minicpm_sala.py`; will spot-check v19's `prepare_model.sh` vs v18 before reusing). Run accuracy + s1/s8/smax on each.

**Before touching fcloud**, I'll extract v19 locally and diff it against v18 so we already know which code paths changed and whether v19 reuses the same preprocessed model. If v19 touched `prepare_model.sh` we'll need to re-quantize. Expected ~5 min of offline diff work.

Baseline decision rule:
- If v19 local accuracy ≥ v18 AND v19 speed ≤ v18 → pick v19 as baseline
- If v19 regresses on either → pick v18
- (We already know v1_007 is slower with no accuracy benefit → won't be baseline)

Then layer Iteration A-0 on top of the chosen baseline for the v20 submission.

---

## Iteration A-0 explained — mcq runaway fix

### The bug (direct evidence)

Today's A/B test confirmed it on both packages:

| | v1_007 mcq | v18 mcq | Expected |
|---|---|---|---|
| avg_in tokens | 270 | 270 | 270 ✓ |
| **avg_out tokens** | **8,505** | **12,267** | **~50** (e.g. "Let me think... ANSWER: B") |
| accuracy | 53.33% | 53.33% | — |

mcq prompts are short (~270 tokens) and should produce a single-letter answer like `ANSWER: B`. Instead the model generates **8,500 to 12,000 tokens of thinking** per mcq question. At 30 questions × concurrency 32, that's hundreds of thousands of wasted generation tokens.

### Why it happens

The eval harness (`eval_model_001.py`, line 352) calls `generate(max_out_len=65536)` for ALL tasks — including mcq — and sets `enable_thinking=True` on the chat template. The server has no `--reasoning-parser`, so the model's `<think>` block runs until one of:
1. Model emits `</think>` → extractor in `eval_model_001.py:178` splits on it and finds the final answer ✓
2. Model hits `max_tokens=65536` → final answer truncated, extractor returns the thinking blob → scorer returns 0 ✗

Sometimes (case 1) we're lucky — mcq scores. Sometimes (case 2) we hit truncation — mcq scores 0. This is why same-package resubmits gave 78.71 / 80.51 / 76.64: the fraction of mcq samples that hit case 1 vs 2 shifts with concurrency, KV pressure, and random scheduling order.

### Why it's a speed killer too

- Each mcq question at 12K output tokens × 30 questions = **360K decode steps**, roughly **equal to one full QA run** — entirely wasted
- Under concurrency 8 or uncapped, long-running mcq chains occupy KV slots and block new prefills → Smax blows up (official saw 2864s)
- If official Smax set has more mcq or harder mcq, the damage scales linearly

### The two-layer fix

**Layer 1 — Eval-side** (local-only, proves hypothesis fast):
- Per-task `max_out_len`: mcq→1024, qa→512, niah→1024, cwe/fwe→16384
- Add `</think>` and `<|endoftext|>` to the stop list
- Doesn't ship to official; just lets us validate that capping mcq length restores accuracy

**Layer 2 — Model/server side** (this is what ships to competition). Four candidates, to test in order of safety:

1. **`--reasoning-parser qwen3` on the server** (~0 risk): makes SGLang split reasoning from content in the API response. If the official scorer uses OpenAI-style `reasoning_content` vs `content` split, this alone fixes extraction for free.
2. **Disable `enable_thinking` in chat_template for short prompts** (low risk): edit `tokenizer_config.json` via `preprocess_model.py` so mcq-shape prompts (<4K in) skip thinking mode entirely. mcq becomes direct-answer.
3. **Add `</think>` to `generation_config.json` `eos_token_id`** (low risk): hard-stops generation when the model emits `</think>`. Works as long as the model emits it — which is the normal training behavior.
4. **Shrink thinking budget via a per-request `max_tokens`** (medium risk): requires custom inference hook; avoided unless 1–3 fail.

### Success metric

Local mcq `avg_out` drops from ~10,000 tokens to ≤1,000 tokens, overall accuracy stays ≥ 78% (ideally rises to 82%+ because currently-truncated mcq answers start extracting correctly), and Smax drops significantly because KV slots free up faster.

### Risk

The big risk is that cwe/fwe (which legitimately produce ~13K output tokens of reasoning) degrade if we over-restrict thinking globally. Mitigation: every fix above is per-task or conditional on prompt length — we never apply a blanket cap.

---

Sound good? If yes, tomorrow's order is:
1. (Offline) diff v19 vs v18 to know what changed
2. (Fcloud) 3-way accuracy+speed, shut down, pick baseline
3. Draft Iteration A-0 implementation on chosen baseline (bilingual CHANGE doc + code)
4. Sync + test + submit as v20Continued with Autopilot: Task not yet completeProvided tomorrow's 3-way test plan (v1_007 / v18 / v19 A/B/C on fcloud with same preprocessed model + symlink swap) and explained Iteration A-0 in detail: the mcq runaway bug (8,505-12,267 output tokens vs expected ~50), why it causes both the accuracy instability (truncated mid-thinking → extractor returns thinking blob → score=0) and Smax blowups (360K wasted decode steps blocking KV slots), and the two-layer fix (Layer 1 eval-side per-task max_out_len + stop-word additions to validate hypothesis; Layer 2 model/server-side with 4 candidate fixes in risk order: `--reasoning-parser qwen3` → conditional thinking disable for short prompts → `</think>` in eos_token_id → per-request max_tokens hook).  


