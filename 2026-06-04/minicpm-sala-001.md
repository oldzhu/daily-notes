Good question — these two numbers describe different layers of the same step, and the WeChat post's "K=1" means exactly what you're asking.

## In our code

- **K = `num_medusa_heads`** = number of **trained Medusa heads** = number of **speculative draft positions** predicted *per step*.
  - K=1 → one trained head → one speculative token predicted per decode step.
  - K=2 → two trained heads → two speculative tokens predicted per step (typically a small tree, not a linear chain).
- **ndt = `draft_token_num` = `num_draft_tokens`** = total length of the **verify input** fed to the target model = `K + 1`.
  - The extra `+1` slot is the **bonus / root** position — the model's own next-token prediction at the current position, which is always going to be correct (target argmax) and serves as the root of the verify tree.
  - K=1 → ndt = 2 (one bonus + one speculative draft).
  - K=2 → ndt = 3 (one bonus + two speculative drafts).

So `--speculative-num-medusa-heads 1 --speculative-num-draft-tokens 2` in our `prepare_env.sh` means: **K=1, ndt=2** — one trained head producing one draft, fed alongside one bonus position into a 2-token verify forward.

## Per-step token count

With K=1 (ndt=2), in the **best case** (draft accepted), one verify step commits:

- 1 bonus token (always correct — the model's own argmax at the current position) **plus**
- 1 speculative token (the draft, accepted because it matched the model's argmax at the bonus's next position) **plus**
- 1 more bonus token for the **next** step is derived from the second forward position's argmax.

That's why Medusa K=1 advertises up to **2 tokens per decode step** when accept_rate is high. In the **worst case** (draft rejected) it commits 1 token (the bonus) — same as standard decode. So K=1 gives at most a **2×** speedup, scaled down by the head's accept rate.

K=2 (ndt=3) can commit up to 3 tokens per step → up to 3× speedup; K=4 (ndt=5) up to 5×.

## Answering the WeChat post

> "Medusa K=1 的 verify overhead 约为 0.39 ms"

Yes — **K=1 there means exactly the same thing**: one Medusa head predicting **one** extra token, verified by a single 2-position forward (ndt=2). The 0.39 ms is the cost of that 2-position verify forward (vs the 1-position cost of a normal decode step). 

The reason that overhead matters: if `verify_forward_time(ndt=2) > 2 × decode_forward_time(ndt=1)`, then even at 100% accept rate K=1 is a net loss. On their hardware they measured 0.39 ms verify overhead — small enough that any non-trivial accept rate is a win.

For our MiniCPM-SALA on SM120 the relevant number isn't on their post; we'd need to measure `verify_forward_time(ndt=2) - decode_forward_time(ndt=1)` locally to know our break-even accept rate.

## Why K=2+ is *not* free

K=2 doesn't just mean "two heads" — it means a **tree** of drafts (typically top-k₁ × top-k₂ branches per head, e.g. 4×2 = 8 candidate sequences). ndt grows like `1 + sum(branch_sizes)`. For the small Medusa-2 paper config, ndt is 26 (1 root + 25 tree nodes). That's a 26-position verify forward instead of 2. The verify overhead grows superlinearly in K when you preserve tree shape. So:

- K=1 is the **easiest win** to chase (small, simple, ndt=2, low overhead).
- K=2+ pays off only if the trained heads are good *and* the tree-mask flashinfer path is efficient.

Our current Medusa scope (post-revert + the proposal) is K=1 only — exactly the regime the WeChat post is talking about. K>1 is deferred to "Stage 4" in CHANGE_0163.  


