Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -20

Ran terminal command:  cd /home/oldzhu/sglang && for i in 1 2 3 4 5; do echo "[pause try $i]"; out=$(python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1); echo "$out" | tail -5; if echo "$out" | grep -q "已暂停\|success"; then echo "OK"; break; fi; sleep 30; done

Instance paused successfully on retry 1. Now researching the article and MTP/Medusa landscape. 



Fetched https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q

Instance paused (success on retry 1). Article fully fetched and analyzed.

---

# Status

- **fcloud paused**: HTTP 200 `任务已暂停` on retry 1 (the 503 was a transient gateway issue, the underlying task was reachable). Billing released.
- **Console JWT**: still good for several days (per pre-test `console-token-info`).
- **No fresh quant was performed** — the SAMPLES=30 sequential test never reached the requant step because the gateway flapped before we could send the command. Plan stays queued for the next session.

---

# Champion article — what they actually did

Champion week-7 ("香草小张"), article URL: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q . Two pillars:

### Pillar A — **NVFP4 + FourOverSix** (we already built this; gave us C=0)
- We tested it across CHANGE_0151 iter-1…iter-7. Result: norm 88.7%, S1 174s (vs GPTQ 110.7s). **Parked.**
- The article confirms only ~40-43% of blocks pick M=4 (matches our `pct_m4 = 43.14%` exactly), so our implementation is faithful. Our gap is elsewhere — likely the `_init_rope` BF16 path on FP4 weights or modelopt's resmooth introducing extra noise. Not worth re-attacking now.

### Pillar B — **Medusa speculative decoding adapted for GLA recurrent state**
- **Not classic MTP/EAGLE.** Medusa = multiple lightweight "heads" stacked on the last-layer hidden state, each predicting position k ahead. No separate draft model.
- Tree-attention verify: top-s candidates per head → Cartesian product → tree → one main forward → longest-accepted-prefix wins.
- The **novel** contribution is **GLA state forking**: in MiniCPM-SALA's hybrid arch, the 24 lightning (GLA) layers carry recurrent state `h_t = exp(−γ)·h_{t−1} + k_t·v_t^T`. Sibling branches of the verify tree must each fork from the **parent's** GLA state, not linearly inherit from the previous candidate. Standard Medusa code in sglang doesn't know about this.
- Reported result: **K=1 verify overhead ≈ 0.39 ms**, decode throughput up across all concurrency levels.
- Training tip: **weight Medusa head training data toward the eval distribution** (qa/mcq/cwe/fwe/niah token statistics); raises accept rate measurably vs random sampling.

---

# Where Medusa hits us in the scoring formula

Score = `S₁·40% + S₈·30% + S∞·30%`, all relative to fastest player.

| Concurrency | Bottleneck today | What Medusa changes |
|---|---|---|
| **S₁** | Decode-bound, batch=1, weight-bandwidth limited (~110.7s) | Each decode step now produces 1 + accepted tokens. K=2 head, accept rate p, multi-token speedup ≈ 1 + p (typical p=0.5–0.7 → **+50% throughput / -33% S1**). **Highest leverage.** |
| **S₈** | Mixed prefill+decode, GEMM kicks in (~40.3s) | Verify forward batches the tree (small extra FLOPs); gain proportional to accept rate but kernels start saturating. Expected −10..20%. |
| **S∞ / Smax** | Compute-bound prefill, GEMM dominant (~32.5s) | Smaller win (extra verify cost competes for the same GEMM kernels). Expected −5..10%, possibly even neutral. |

**S₁ is exactly where Medusa shines and where our 40% weight lives.** The champion's win matches this asymmetry.

---

# Other MTP-family options for context

| Method | Draft cost | Adaptation effort on SALA | Best concurrency | Verdict for us |
|---|---|---|---|---|
| **Medusa** (article's choice) | Tiny — 2-3 MLP heads on h_t | Medium — must add GLA state-fork in tree verify | S1 ★★★, S8 ★★, Smax ★ | **Recommended.** Highest S1 leverage; sglang already has a Medusa worker scaffold to start from. |
| **EAGLE / EAGLE-2 / EAGLE-3** | One small autoregressive layer + classifier; trains a draft head | Higher — needs training a real "tiny SALA" draft, plus the SAME GLA-state-fork problem | S1 ★★★ | We tested EAGLE3 on Test 22 with random draft → accept_rate=0.26, S1 +65%, killed. A *trained* EAGLE head could match Medusa, but training cost + tarball size > Medusa heads. |
| **MTP (Multi-Token Prediction, Meta-style / DeepSeek-V3)** | Drafts produced by extra transformer blocks at the end of the main model; trained with auxiliary loss | High — same GLA fork problem + need to plumb extra blocks through compile | S1 ★★ | Heavier than Medusa, comparable to EAGLE. Not a clear win over Medusa here. |
| **Lookahead / SpS (n-gram)** | No training | Low | S1 ★★ | Would need to fit the SOAR private set's token distribution; risky on niah/cwe. Cheap to spike but unpredictable. |
| **Self-speculative (pruned base model)** | Use the model itself with fewer layers as draft | Medium-high (must skip lightning layers safely → state divergence) | S1 ★★ | The lightning layers themselves dominate cost; skipping them changes outputs. Risky. |

For our pipeline, the dominance ranking is clear: **Medusa > EAGLE-trained > MTP > Lookahead > self-spec**.

---

# Our current state (per memory + last test)

- Best config: GPTQ sparse_qkv_w8 + FP8_e5m2 KV + dense + Tier1 long-ctx + flashinfer + torch.compile bs=24.
- Today's retest: **ori_acc=77.47% / norm=96.83% / C=0** (single-run noise, not a real regression).
- Speed: S1=110.7 / S8=40.3 / Smax=32.5 — already 9% faster than the v18-A submission baseline.
- The accuracy gap to C=1.0 is the **bigger** open problem than speed (norm needs +0.17pt to clear 97%, +2.17pt to clear 99%).

---

# Proposed next-iteration plan

### Phase 1 (queued, cheap) — finish the GPTQ calibration sweep
The plan we already approved: re-quantize `MiniCPM-SALA-30-sequential-sparse_qkv_w8` and rerun accuracy + speed. This is **0 risk, 0 source change, ~1.5 h fcloud time**. Mirrors the NVFP4-FOS-5 finding (sequential ≪ stratified) on the GPTQ side. Do this when fcloud is back.

### Phase 2 (medium effort, big upside) — **Medusa-on-SALA** following the champion recipe

Why Medusa beats every other speed lever in our catalog right now:
- Marlin GEMM tile work (CHANGE_0125) → **neutral** historically.
- KV-cache compression → **neutral** because we're already FP8 and KV bandwidth isn't the S1 bottleneck.
- W4A8 / FP8-blockwise GEMM → already tried, **regression** (CHANGE_W4A8_001: S1 +118%).
- Sparse path → still hangs on HEAD (Round 13d).
- Medusa K=1 / K=2 directly attacks S1, which has **40% of the score**.

**Sub-tasks (rough sizing only — full proposal will follow your green-light):**
1. **R1 — Code spike (no training).** Wire sglang's existing `srt/speculative/eagle_worker.py` + `srt/models/minicpm_eagle3.py` skeleton into a Medusa-shaped K=2 path with **untrained random heads** + GLA state fork. Smoketest: does the tree-verify forward run end-to-end on SALA without crashing? *Goal: prove the plumbing.*
2. **R2 — GLA state fork.** Locate every recurrent-state read inside `srt/layers/attention/fla/` + `hybrid_linear_attn_backend.py`. Add a "branch_id" axis to the GLA state buffer; before each verify forward, broadcast parent state to all sibling branches. *Goal: correctness.*
3. **R3 — Train Medusa heads.** ~2-3 small MLP heads (≈ 30-100 MB total — fits 2 GB budget). Calibration corpus = stratified mix of public eval distribution (qa/mcq/cwe/fwe/niah). Train on a single H100 / RTX 6000 for a few hours. Submission tarball gets the head weights (Apache-2.0 license, our own training).
4. **R4 — Bench.** Target accept_rate ≥ 0.5 at K=2 → S1 expected ≈ 75-85s (vs 110.7s today, **−25-32%**).

### Phase 3 (small wins, fast) — guard accuracy
- Verify mcq runaway-think mitigation again on the new fcloud (avg_out=11086 in today's run vs 8505 on Test 12 → MCQ gen is still pathological).
- Run a 2-shot variance probe on today's config to lock down the ±2.5pt floor before committing to any new "win" measurement.

---

# Risk / rule-compliance check for Phase 2 (Medusa)

- **Submission size**: Medusa heads are compact (≪ 100 MB); 2 GB budget intact.
- **On-site quantization rule**: heads are weights, but they are **trained**, not quantized — competition rules allow trained speculative heads (mentioned explicitly: "Speculative heads allowed (count toward 2GB)").
- **5h on-site time**: head training is offline; only quant + eval count. No impact.
- **Apache 2.0**: our heads + our training script — clean.
- **Accuracy**: Medusa is **lossless when implemented correctly** (verify rejects bad drafts, falls back to base model). Risk concentrated in the GLA-fork bug surface.
- **Code complexity**: this is the largest source change we've planned. Need careful staging.

---

# Recommendation

1. **Now (offline)**: I write a full bilingual proposal `PROPOSAL_medusa_minicpm_sala_001.{en,zh}.md` with file-level edit plan, GLA-fork pseudocode, training-data curation strategy, and a phased validation matrix. **Awaiting your "write the proposal" go.**
2. **When fcloud is back**: run the queued SAMPLES=30 sequential test first (0 source change, 0 risk) → adds a data point to TEST_RESULTS_TRACKING and lets us confirm the new fcloud is healthy.
3. **Then**: kick off Medusa Phase 2 R1 (code spike only, no training yet) on a non-fcloud machine if we have GPU; otherwise schedule for the next fcloud session.

Tell me which to start: **(a)** write the Medusa proposal now, **(b)** also do a brief follow-up survey of EAGLE-3 and DeepSeek-MTP variants for a side-by-side comparison doc first, **(c)** queue both and wait for fcloud first?
