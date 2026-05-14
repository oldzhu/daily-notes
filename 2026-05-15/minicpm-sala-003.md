## v20 vs v18 — quick read

| | acc_ori | acc (norm) | C | S1 | S8 | Smax | final |
|---|---|---|---|---|---|---|---|
| prev-18 | 79.24 | 99.06 | 1.0 | 596.25 | 1066.35 | 2746.62 | **39.25** |
| v18 | 80.13 | 100.0 | 1.0 | 586.64 | 1087.9 | 2857.64 | **39.01** |
| v19 | 78.07 | 97.58 | 0.96 | 591.04 | 1113.65 | 2917.74 | **35.37** |
| v20 | 80.87 | 100.0 | 1.0 | 586.08 | 1080.25 | 2839.78 | **32.84** |

v20 has the **best accuracy ever** but the **worst final_score** — purely because every other team got faster, so our relative S₁/S₈/S∞ % dropped. C=1.0 already; accuracy improvements no longer move the needle. Agree to pause accuracy work and pivot to speed.

---

## What "v2 fix" is

**Goal:** suppress reasoning/thinking tokens **only on mcq prompts** so mcq answers don't burn 10k+ output tokens (which both wastes time and truncates the final answer when it hits max_tokens). Non-mcq tasks (qa/niah/cwe/fwe) keep thinking enabled.

**Files changed (commit `0c7767aa8`):**
- preprocess_model.py — patches the model's `chat_template.jinja` at packaging time
- toggle_mcq_thinking_patch.py — fcloud A/B helper that flips an already-quantized model in place

**Why v1 was a no-op:** v1 prepended a preamble that set `enable_thinking=false` for mcq. But `grep enable_thinking` in the model's `chat_template.jinja` returned **zero gating uses** — the template never reads that variable. The model is trained to always start the assistant turn with `<think>`, so v1 changed nothing. A1 confirmed: mcq=56.67%, avg_out_len=10438.

**How v2 works (Qwen3 standard "disable-thinking" trick):**

The template ends with this block, which simply emits the assistant header:
```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

v2 **replaces** that exact block with:
```jinja
{# SOAR_MCQ_THINKING_DISABLE_v2 #}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- set _soar_mcq_ns = namespace(disable_think=false) -%}
    {%- if messages is defined and messages -%}
        {%- for _m in messages -%}
            {%- if _m['content'] is string and 'LETTER is one of ABCD' in _m['content'] -%}
                {%- set _soar_mcq_ns.disable_think = true -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if _soar_mcq_ns.disable_think -%}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif -%}
{%- endif %}
```

Mechanism, step by step:
1. **Detection** — scan all messages for the literal `"LETTER is one of ABCD"` (present in 30/30 public mcq prompts, 0 false positives across qa/niah/cwe/fwe).
2. **Pre-seed** — if mcq, after emitting `<|im_start|>assistant\n` the template **also emits a closed `<think>\n\n</think>\n\n`**. The model sees its `<think>` slot already opened-and-closed, so its first sampled token is the **answer**, not more reasoning.
3. **Non-mcq is untouched** — header is emitted, model freely opens `<think>...` as trained.
4. **Jinja namespace()** is used because `{% set %}` inside a `{% for %}` is block-local; namespace mutation is the standard Jinja workaround for cross-block state.

**Local Jinja2 render tests (passed before push):**
- mcq → `<|im_start|>assistant\n<think>\n\n</think>\n\n` ✓
- qa → `<|im_start|>assistant\n` (clean) ✓
- "Pick A or B" without exact trigger → no false positive ✓
- apply / idempotent / revert / legacy-v1-cleanup helpers ✓

**What's NOT yet done:** v2 has only been validated in local Jinja render. It has NOT been run on fcloud — A1/A2/S1 with v2 are still pending. Given your decision to pause accuracy work, we can either:
- (a) shelve v2 entirely (v20 already has acc 100% norm, C=1.0 — accuracy is solved), or
- (b) keep the code in place but skip applying it for now, since it's safe (no-op when env `SOAR_DISABLE_MCQ_THINKING=false` or when toggle is off).

Recommendation: **(b)** — leave the code committed but don't apply it on the model. It costs nothing and is ready if a future submission ever drops below C=1.0.

Ready to discuss the **speed pivot**. Want me to pull up the optimization catalog (OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md) and rank what's left?  


