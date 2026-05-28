Starting Stage 3. Let me first review the current MedusaWorker code and the existing Stage 3 design notes from CHANGE_0153/0154/0155 to inform the proposal. 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)

Searched for files matching `**/minicpm_medusa_heads.py`, 1 match

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm_medusa_heads.py)

Let me check what sglang already provides for the Medusa verify path before designing.  

Searched for regex `TARGET_VERIFY|target_verify|forward_mode.*verify` (`**/python/sglang/srt/**/*.py`), 40 results

Excellent — `minicpm_backend.py:521` already has a TARGET_VERIFY branch (our attention backend supports verify), and ngram_worker.py is the simplest existing reference (linear K draft tokens, no tree). Let me study it.  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py)

Let me check the hidden state capture mechanism for Medusa-style speculation.   

Searched for regex `capture_hidden_mode|CaptureHiddenMode` (`**/python/sglang/srt/**/*.py`), 25 results

Now let me draft the Stage 3 proposal documents (bilingual), then implement.    

Searched for regex `class NgramVerifyInput|def verify|def prepare_for_verify` (`**/python/sglang/srt/speculative/ngram_info.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_info.py#40-40), lines 40 to 200

Let me commit a clear plan via todo and proposal docs first, then implement.  

Updated todo list

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.zh.md)

Let me commit the proposal and pause briefly for your review — the next step is substantial code (hidden-state plumbing + verify wiring), worth ~5 min of your reading time to course-correct before fcloud burn.   

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.en.md docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.zh.md && git commit -q -m 'docs(medusa-r1b): Stage 3 proposal (verify + rewind, eager 3a → cuda-graph 3b)' && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Proposal pushed (`5296852ac`):
- PROPOSAL_medusa_stage3_verify_rewind.en.md
- PROPOSAL_medusa_stage3_verify_rewind.zh.md

## TL;DR of the proposal

**Plan**: Two sub-stages.

| Stage | Mode | Goal | Time |
|-------|------|------|------|
| **3a** (this session) | EAGER (cuda-graph OFF via existing `SOAR_SPEC_MEDUSA_EAGER=1` toggle) | Wire heads + verify + hidden-state capture. At zero-init heads ⇒ 100% accept ⇒ **byte-identical output**. Speed expected slower than baseline (eager + verify overhead). | ~half session |
| **3b** (next session) | cuda-graph ON | Capture TARGET_VERIFY in cuda-graph; measure real speed ceiling. | next iter |

**Key insight**: ngram_worker.py + ngram_info.py already implement the full linear-K-token verify pipeline on this exact server config. Our `minicpm_flashinfer` backend has a `is_target_verify()` branch at L521. We **reuse `NgramVerifyInput` verbatim** and just replace n-gram lookup with `MedusaHeads.forward(prev_hidden).argmax`.

**Why correctness is free at Stage 3a**: With `W1=0`, `MedusaHead(h) = SiLU(0) + h = h`, so `argmax(MedusaHeads(h)) = argmax(target_lm_head(h))` = exactly the target's argmax. Verify always accepts → output byte-equal to baseline. We sidestep the GLA-rewind risk because rewind never fires.

**Three risks worth flagging**:
1. NGRAM TARGET_VERIFY untested on our current FP8-KV + dense + mixed-chunk config — could fail at runtime.
2. Hidden-state capture needs threading through the model_worker_batch → forward_batch → return path. Easy to silently drop.
3. NgramVerifyInput type-tag (`SpecInputType.NGRAM_VERIFY`) — if any downstream checks the type strictly we may need to subclass.

**My recommendation**: proceed with Stage 3a implementation now. If anything looks broken in local sanity, I report back before burning fcloud minutes.

**Want me to**: (a) **proceed to code 3a now**, or (b) you want to read the proposal first and then say go?

(fcloud is currently paused, so reading time costs nothing.)  



