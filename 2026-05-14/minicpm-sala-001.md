Searched for regex `extract_final_answer|</think>|mcq|score` (`**/benchmark/soar/demo_sala/eval_model_001.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#170-170), lines 170 to 210

**Yes, exactly right.** I just traced it through the harness — disabling thinking for mcq is fully compatible with the extractor. Here's the trace:

### Code path (eval_model_001.py:176-201)

```python
def extract_final_answer(pred):
    parts = pred.split('</think>')
    return parts[-1].strip() if len(parts) > 1 else pred   # ← key line

def extract_mcq_answer(pred):
    match = re.search(r'(?i)ANSWER\s*:\s*([A-D])', pred)
    if match: return match.group(1).upper()
    # ... also accepts \boxed{B} and \boxed{\text{B}}
```

### Case 1: thinking ON, lucky run (current state)

Model output: `<think>...</think>\nANSWER: B`
- `split('</think>')` → `["<think>...", "\nANSWER: B"]` (length 2) → returns `"ANSWER: B"`
- regex matches → `"B"` ✓

### Case 2: thinking ON, unlucky run (current failure)

Model output: `<think> reasoning reasoning reasoning ... [10,946 tokens, never closes]`
- `split('</think>')` → `["<think> reasoning..."]` (length 1) → **falls through to `else`, returns the whole blob**
- regex searches the whole thinking blob → may match a stray "ANSWER:" mention inside reasoning, or fail entirely → mostly 0

### Case 3: thinking OFF (Phase 14.1 target) ✓

Model output: `ANSWER: B`
- `split('</think>')` → `["ANSWER: B"]` (length 1) → falls through to `else`, returns `"ANSWER: B"` as-is
- regex matches → `"B"` ✓

**Same correct result as Case 1, but deterministic.** The `len(parts) > 1` branch was never the *required* path — it was the *safe* path; the `else` branch is also a safe path **when there is no thinking blob to begin with**. The runaway problem is specifically the case where there IS a thinking blob but it doesn't close.

### Bonus robustness

The mcq regex accepts:
- `ANSWER: B` (most common, what we'll target)
- `answer: b`, `Answer:B`, `ANSWER  :  B` (case + whitespace tolerant)
- `\boxed{B}` and `\boxed{\text{B}}` (math-mode variants)

So even if the model occasionally emits a slightly different format with thinking off, the extractor still finds the letter.

### Implication for Phase 14.1 design

Because the no-thinking output is the simplest case for the extractor, the chat-template patch can be very minimal:

1. For mcq, prepend `enable_thinking=false` (or the model's equivalent control token) so the model goes straight to "ANSWER: X".
2. Optionally, **also** add a system instruction like "Answer in the format `ANSWER: X` where X is the letter." to nudge the model toward the regex's preferred form. (Optional — the model already does this most of the time.)

No harness change required. No risk to other tasks. Confirmed compatible.

If this also satisfies your sanity check, I'll proceed to draft `CHANGE_0140_mcq_thinking_disable.{en,zh}.md` with the exact `preprocess_model.py` Jinja patch + a small offline test script that re-scores existing `predictions.jsonl` to estimate the lift before we burn an fcloud round.  


