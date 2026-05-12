Yes — **scope is mcq only**. Other tasks keep thinking enabled. Reasoning:

| Task | Current acc | Avg out tokens | Thinking helps? | Decision |
|---|---|---|---|---|
| **mcq** | ~55% (40–63 swings) | 10,946 ❌ | **Hurts** — bimodal failure on `</think>` emission | **Disable** |
| qa | ~47% | 104 | Likely helps (multi-step reasoning over long context) | Keep |
| niah | 100% | 360 | Already saturated; don't risk regression | Keep |
| cwe | ~85% | 11,350 (within budget) | Helps (analysis tasks) | Keep |
| fwe | ~99% | 13,806 (within budget) | Helps | Keep |

Only mcq has the pathology where (a) the answer is a single letter so thinking adds zero value, and (b) the `</think>` extractor split makes scoring depend on emission of that specific token. The other four tasks either tolerate or actively benefit from chain-of-thought.

### Mechanism (how the chat template knows it's mcq)

Two viable detection paths — we'll pick whichever the actual MiniCPM-SALA chat template supports cleanest. I'll confirm by reading the model's `tokenizer_config.json` on fcloud during CHANGE_0140 drafting:

1. **Sampling-parameter signal** (preferred if available): the harness passes `chat_template_kwargs={"enable_thinking": True}` unconditionally, but our patched template reads an **additional** kwarg like `task` or detects via the message structure. Then in `preprocess_model.py`'s chat-template patch we set `enable_thinking = false if (system_prompt contains "Choose the correct answer" or messages[-1].content matches mcq heuristic) else true`.

2. **Content-pattern signal**: the system prompt or user prompt contains a stable mcq marker (`"A.", "B.", "C.", "D."` lines, or the literal phrase `"Choose the correct answer"` from the public set). Template Jinja inspects the last user message and flips `enable_thinking` based on a regex/substring check.

Path 2 is most robust because it doesn't rely on the harness sending us a flag. The marker pattern is stable across public+private (mcq tasks always present 4 lettered choices).

### Conservative default

If the detection logic ever **misclassifies** (e.g., a non-mcq sample happens to contain "A.", "B." in its content), the consequence is "we disabled thinking on a non-mcq sample" — which is at worst neutral on niah (it's pure retrieval) and a small per-sample regression on qa/cwe/fwe. Magnitude estimate: ≤ 1pt total even with 5–10% false-positive rate. Compared to the +7pt mcq upside, this is a clearly favorable trade.

### What CHANGE_0140 will spell out

When I draft the change doc, I will give you the exact Jinja diff (something like):

```jinja
{%- set is_mcq = ... -%}
{%- if is_mcq -%}
  {%- set enable_thinking = false -%}
{%- endif -%}
```

…plus the precise detection predicate, validated by inspecting the actual chat-template string and a few sample prompts on fcloud. You'll see the exact predicate before any code lands.

Please confirm: **mcq-only scope, with content-pattern detection (path 2), conservative default of leaving thinking ON when detection is ambiguous.** If you agree, I'll proceed to draft CHANGE_0140.  


