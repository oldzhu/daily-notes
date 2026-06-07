## What "iteration" means here

A single generate request goes through many **scheduler steps**. For a 65k-token mcq output, the engine runs roughly:

- 1 prefill step (sometimes chunked into a few)
- ~5,000–60,000 decode steps (one per token group emitted)

With spec decoding, each decode step is actually a **verify step**: draft K tokens → forward target model on K candidates → accept some prefix → commit. That whole cycle = **one iteration**.

So "iteration 1" is the first verify cycle after the prompt is ingested. "Iteration 2" is the next one. "Iteration 100" is deep into decoding. The model's state (KV cache, GLA recurrent state, sampler state) evolves continuously through these.

## What the current preflight does (1-iteration)

`preflight_medusa_vs_ngram.py` does this:
1. Start the engine in **NGRAM** mode, feed one fixed prompt, snapshot 25 tensor fields at 4 points (`pre_prepare_for_verify`, `pre_verify`, `post_forward`, `post_verify`) of the **first** verify call, then stop.
2. Restart the engine in **MEDUSA** mode (K=1, `head_pred=0` so draft is byte-identical to ngram's), feed the same prompt, snapshot the same 25 fields at the same 4 points of the **first** verify call.
3. `preflight_diff.py` compares the two pickles field-by-field.

After iter 4 fix: every byte matches. We declared victory and ran the full eval. Result: garbage.

## Why 0-diff on iter 1 is not enough

The diff only proves: *given the same prompt + the same initial state, after exactly one verify cycle, NGRAM and MEDUSA produce the same observable tensors at those 4 boundaries.*

It does **not** prove any of these:
- That **iteration 2** sees identical inputs. (If iter 1 had some side-effect that lives outside the 25 dumped fields — e.g. the GLA recurrent state cache, an internal `req_to_token` pointer, a scheduler accounting field — iter 2's verify starts from a slightly different state, and the divergence snowballs.)
- That the verify **kernel** chose the right token. (We dump `next_token_ids` = the integer that came out, but if the kernel picked it from logits index `i` in MEDUSA mode vs index `j` in NGRAM mode and the two happened to alias to the same token id on this particular prompt's first iteration by coincidence, we'd see "match" while the underlying selection rule is broken.)
- That **DECODE** in eager mode (the actual hot path during the eval) behaves identically. Preflight uses the engine's normal warmup → first verify is on a small batch. Eval runs at batch sizes 1..24 with cuda-graph DECODE bypassed (`cuda graph: False`) for the MEDUSA path. A different code path means a different bug surface — not covered.

These three are exactly the H1 / H2 / H3+H4 hypotheses in §5.7.

## What "multi-iteration preflight" would do

Same setup, but dump at iterations 1, 2, 5, 10, 25 (and ideally also at the prefill→decode boundary, and after the first chunk boundary in chunked-prefill). Then diff each iteration independently.

Outcome possibilities:
- All match → the divergence is *inside* one verify cycle on a path we don't dump (i.e. inside the kernel — see "verify tracer" below) or in something we never read (sampler RNG, KV slot indices).
- iter 1 matches but iter 2 diverges → the bug is in the *transition* between iterations. Most likely culprit: GLA recurrent state not being restored or being advanced by the wrong amount after a partial accept. Diff'ing the specific tensors that differ at iter 2 (e.g. `gla_state[layer_i]`) pinpoints the exact tensor.
- iter 1..5 match, iter 10 diverges → cumulative drift in a continuous value (FP8 KV quant error, GLA hidden state drift). Different fix class: numeric tolerance, periodic resync.
- iter 1 already diverges in DECODE (not EXTEND) → preflight should also cover the first DECODE step, which our current one doesn't. The eager-mode decode path (CHANGE_0157's `is_target_verify` branch in `SimpleGLAAttnBackend.forward`) becomes prime suspect.

So multi-iteration preflight is a **localization tool**: it tells you *which iteration first diverges*, and *which field* diverges first. That's almost always sufficient to read the relevant code path and identify the bug.

## What "in-kernel verify tracer" would do

The kernel that runs verify is in `python/sglang/srt/layers/attention/...` plus the `spec_info.verify()` walk that consumes logits and decides accept length. The current preflight dumps **inputs** (positions, tree_mask, retrive_*) and **outputs** (next_token_ids, accept_length). It does **not** dump intermediate state inside verify:

- which row of `logits` was read for each candidate
- which token won the greedy/sample step
- which path through the accept tree was taken
- the boolean mask of accepted positions

A tracer adds `print(...)` (or a side-channel dump) at three or four points inside `spec_info.verify` and inside the attention kernel's logits gather:
1. The raw logits tensor shape + a few sample rows
2. The argmax per candidate position
3. The accept walk's per-step decision
4. The final `accept_length` and committed token ids

Run it once with MEDUSA, once with NGRAM, diff the two traces. If iter 1's *outputs* match but the *traces* differ, you found the bug — the two paths reach the same answer by different routes, and at iter 2 (when one of those routes' side-effects matters) they diverge.

## Difference between the two and what each catches

| Tool | What it catches | What it misses |
|------|-----------------|----------------|
| 1-iter preflight (current) | Mismatched setup or first-verify outputs | Anything that diverges after iter 1, or that converges in outputs but diverges in internal state |
| Multi-iter preflight | The exact iteration where state first diverges; which **field** drifts first | Bugs purely inside the kernel that don't show up in any dumped field |
| In-kernel verify tracer | Bugs in the verify decision path even when inputs and outputs both match | Bugs in non-verify code (KV slot bookkeeping, GLA state advance, scheduler) |

They are **complementary**. Multi-iter preflight tells you "iter 7 is where it breaks, in field X." Then you look at the code that writes X between iter 6 and iter 7. If that code reads from the verify kernel, you add the in-kernel tracer to see why X got the wrong value.

## Will multi-iter preflight find and fix the cause completely?

**Find: very likely yes.** Once you know "iter N diverges first in field F," the surface area is tiny — usually one function. The H1–H4 hypotheses in §5.7 are exactly the four shapes the divergence can take, and each one shows up at a different iteration/field signature.

**Fix completely: probably yes, but with a real risk of "whack-a-mole."** Spec decoding has multiple coupled state machines (KV pool, GLA recurrent state, scheduler accounting, sampler). Our CHANGE_0157 / 0158 / 0159 / 0160 / 0161 chain already shows that "fix one symptom, surface another" is common here. A realistic estimate:

- 60–70% chance multi-iter preflight + targeted fix lands at C=1.0 in one or two more iterations.
- 30–40% chance we find one root cause, fix it, get further into decoding, then hit a second deeper bug (e.g., FP8 KV + GLA quant drift) that needs its own round.

That's why §5.7 also lists option (3) **abandon Stage 3a and ship Stage 2-cgraph as v23** — it's already C=1.0 at 80.11% with a small S1 speedup, and locks in submission safety while leaving Stage 3a as a future optimization. The choice depends on how much submission risk-margin you want vs. how much time you want to spend chasing the verify bug.
