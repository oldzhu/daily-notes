You're exactly right, and I should drop Layer 1.

**Your reasoning is correct:**
- `eval_model_001.py` is our local copy of the official harness. It is NOT in the submission package.
- Official eval uses their own unmodified version with `max_out_len=65536` + `enable_thinking=True`.
- If we modify our local copy (per-task caps, extra stops), we gain nothing officially — and we **lose** our local↔official signal correlation. A passing local test would no longer predict official behavior.

**So Layer 1 should be dropped from the plan.** I was using it only as a "hypothesis confirmation shortcut," but that's unnecessary — we can confirm the hypothesis just as well by applying a Layer 2 fix and observing mcq `avg_out` drop on the **unmodified** eval script. That's actually a cleaner signal because it's identical to what official will see.

**Revised Iteration A-0 plan:**

Skip Layer 1. Go straight to Layer 2, validated against pristine `eval_model_001.py`:

1. **Pre-step**: keep our local `eval_model_001.py` **byte-identical** to the official version. No changes ever. If we need to inspect predictions for debugging, do it in a separate scratch script that reads `outputs/<timestamp>/predictions.jsonl` — never touch the harness itself.

2. **Layer 2 fix** (ships to official, tested locally):
   - **Candidate 2.1**: `--reasoning-parser qwen3` in `SGLANG_SERVER_ARGS` — zero risk, free trial
   - **Candidate 2.2**: add `</think>` to `eos_token_id` in `generation_config.json` via `preprocess_model.py`
   - **Candidate 2.3**: conditional `enable_thinking=False` via chat_template edit for short prompts
   
3. **Validation** (same eval script as official):
   - Success: mcq `avg_out` drops from ~12K → ≤1K AND overall accuracy ≥ 78% AND Smax drops
   - Failure: either metric worse → revert, try next candidate

4. **Submission**: once a candidate passes, package as v20 — the fix affects official because it lives in `prepare_env.sh` / `generation_config.json`, both of which ARE in the submission tarball.

**Also revising tomorrow's 3-way test**: same mechanism (v1_007/v18/v19 symlink swap), all three use the **unmodified** `/root/data/eval_model_001.py`, so the comparison is apples-to-apples with official.

One action item to enforce this: I should add a check/rule somewhere so nobody ever modifies the eval script. Want me to add a "Never modify eval_model_001.py — it must stay byte-identical to the official copy" rule to copilot-instructions.md tomorrow when we start working? 


