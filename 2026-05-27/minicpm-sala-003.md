Starting on **Phase R1 (plumbing spike)** per the proposal — random heads, gated by `SOAR_SPEC_MEDUSA=0` default, with `accept_threshold=1.0` byte-identity as the correctness gate. Let me first map sglang's existing speculative scaffolding so the new Medusa worker mirrors the conventions.  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/spec_info.py#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/base_spec_worker.py)

Searched for regex `minicpm.*sala|SALA|hybrid_linear_attn` (`**/python/sglang/srt/models/**`), 40 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/spec_info.py#80-80), lines 80 to 126

Before writing 1000+ lines I want to do a focused read of the actual GLA recurrence path and the EAGLE worker, then produce an **R1 design doc** with concrete class signatures + diff plan. That's the highest-leverage thing I can do in one turn without risking a half-baked GLA-fork. Let me read the key files in parallel.   

Searched for regex `SimpleGLA|fused_recurrent|chunk_simple_gla|recurrent_state|GLA` (`**/python/sglang/srt/models/minicpm.py`), 20 results

Searched for regex `class |def |recurrent_state|simple_gla|forward_decode|forward_extend` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 20 results

Searched for regex `^class |^    def |verify|draft` (`**/python/sglang/srt/speculative/eagle_worker.py`), 80 results

The EAGLE worker alone is ~700+ lines and the speculative subsystem (workers + cuda graph runners + draft utils + spec_info datastructures) is ~5000 lines. To do this right I need to produce a concrete R1 **design doc** with class signatures and exact diff plan before writing code, otherwise we risk a half-baked GLA-fork. Let me read the most critical pieces — the GLA recurrent state path and the verify-output flow — then write the design doc.  

Searched for regex `^class |^    def |last_recurrent_state|state_indices_buffer|forward_decode|forward_extend` (`sglang` · `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 84 results

Searched for regex `^class |^def |tree_mask|verify|accept|build_tree` (`sglang` · `python/sglang/srt/speculative/eagle_utils.py`), 67 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1440-1440), lines 1440 to 1703

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1373-1373), lines 1373 to 1440

This is a major discovery: `update_mamba_state_after_mtp_verify` (line 1373) already implements **post-verify state scatter** for Mamba2/GDN — `intermediate_ssm` buffer holds K candidate states, then a scatter picks the accepted-prefix state. The pattern just needs to be ported to `SimpleGLAAttnBackend` + adapted to Medusa's tree topology. The "GLA-fork problem" is mostly already solved upstream; we don't have to reinvent it. Let me capture this in a tight R1 design doc.   

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0153_medusa_phase_r1_design.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0153_medusa_phase_r1_design.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0153_medusa_phase_r1_design.en.md docs/soar_2026_changes/CHANGE_0153_medusa_phase_r1_design.zh.md && git commit -m "docs(medusa): R1 design — port mtp state-scatter to SimpleGLA, K=1 chain spike" && git push minicpm-src mixed_minicpm_cudagraph

Pushed `2d658a868`. **Stopping here for your sign-off** — the project's mandatory rule is "Wait for explicit user approval before editing code", and the design doc surfaced two material findings that should be reviewed before the ~635 LOC code phase begins.

### Summary

CHANGE_0153_medusa_phase_r1_design.en.md / .zh.md

**Key discoveries that simplify R1:**
1. The "GLA-fork problem" already has a working solution upstream: `update_mamba_state_after_mtp_verify` (hybrid_linear_attn_backend.py L1373) does post-verify state scatter for Mamba2/GDN. We **port that pattern** to `SimpleGLAAttnBackend` instead of designing from scratch.
2. EAGLE's `build_tree_kernel_efficient` + `verify_tree_greedy_func` are draft-source-agnostic — fully reusable for Medusa.
3. With `W1=0` init, every Medusa head ≡ base model output, so `accept_threshold=1.0` byte-identity gate trivially passes — gives us a clean correctness oracle.

**R1 footprint:** ~635 LOC across 3 new files + 7 modified files (down from the proposal's 1500–2000 estimate).

### Three decisions I need from you (doc §8)

- **Q1.** Proceed to write R1 code now (1–2 turns), or you review the design first?
- **Q2.** Default K = 1 (chain, smallest blast radius — recommended) or K = 2?
- **Q3.** Also add `SOAR_SPEC_NGRAM=1` opt-in to `prepare_env.sh` as free-insurance to test in parallel? Pure server-arg, zero risk. 



