## Yes, only for mcq

The patch fires only when the literal substring `"LETTER is one of ABCD"` is found in any user message. That phrase appears in **30/30 public mcq prompts** and **0 of qa/niah/cwe/fwe**. So qa/niah/cwe/fwe go through the original prompt path unchanged — they keep thinking enabled.

(We chose mcq because it's the only task where thinking *hurts*: long reasoning eats the `max_tokens` budget and the final letter answer gets truncated → 0 score. For qa/niah/cwe/fwe, thinking helps accuracy and the answers are short enough to fit.)

## Why pre-seeding `<think>\n\n</think>\n\n` stops thinking

This is the official Qwen3 "disable thinking" trick. To understand it, you need to know what the model was trained to do:

**During training**, every assistant turn looks like:
```
<|im_start|>assistant
<think>
  ... internal reasoning ...
</think>

  ... final answer ...
<|im_end|>
```

The model learned a strict pattern:
1. After `<|im_start|>assistant\n`, the **next token is always `<think>`**.
2. Inside `<think>...</think>`, freely reason.
3. After `</think>\n\n`, emit the **final answer** (concise, no more reasoning).
4. End with `<|im_end|>`.

**Without our patch**, the prompt fed to the model ends at:
```
<|im_start|>assistant\n
                       ↑ model samples here → emits "<think>" → reasons for thousands of tokens
```

**With our patch (mcq only)**, the prompt fed to the model ends at:
```
<|im_start|>assistant\n<think>\n\n</think>\n\n
                                              ↑ model samples here
```

From the model's point of view, the `<think>` slot is **already opened and closed**. It looks like a turn where the model already finished thinking (with empty content). So pattern-matching on its training, the very next token it samples is the **final answer** (the letter "A"/"B"/"C"/"D"), not another `<think>`.

The model is *not* explicitly told "don't think." We just trick it by putting it in the post-`</think>` state from the start.

## Why this is safe and reliable

- The empty `<think>\n\n</think>\n\n` pattern is what Qwen3 (and MiniCPM-SALA, which inherits Qwen-style training) has seen as a **valid** assistant turn shape during training, just with empty thinking content.
- The model's "answer mode" tokens after `</think>\n\n` were trained to be concise and on-format — exactly what mcq scoring needs (`ANSWER: <LETTER>`).
- It cannot re-open `<think>` after `</think>` — that pattern doesn't exist in training data. So once we close `<think>`, thinking is over for this turn.

## Why the v1 approach failed

v1 tried to set `enable_thinking=false` as a Jinja variable. That works on the *upstream* Qwen3 chat template, which has guards like:
```jinja
{% if enable_thinking %}<think>...{% endif %}
```

But MiniCPM-SALA's `chat_template.jinja` has **no such guard** — it just emits the raw structure unconditionally. So setting the variable did nothing. v2 doesn't fight the template; it changes the **token sequence** the model receives, which is what actually controls behavior.

## TL;DR

- mcq-only, by literal-string detection
- `<think>\n\n</think>\n\n` makes the model's training pattern think the thinking phase is already done, so it samples the answer directly
- It's a **prompt-state trick**, not a flag — that's why it works where v1 didn't  


