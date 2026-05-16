Created 7 todos



Fetched 2 resources

Got the champion post and leaderboard. Now start fcloud to package v22. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && for i in $(seq 1 30); do out=$(python3 scripts/fcloud/fcloud_exec.py exec "echo ready" 2>&1); if echo "$out" | grep -q ready; then echo "READY ($i)"; break; fi; sleep 15; done && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "cd /root/submission_sim && tar --exclude='__pycache__' --exclude='*.pyc' -czf /root/minicpm_sala_submit_v22.tar.gz *.whl *.sh *.py perf_public_set.jsonl sglang && ls -la /root/minicpm_sala_submit_v22.tar.gz && sha256sum /root/minicpm_sala_submit_v22.tar.gz" 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "ln -sf /root/minicpm_sala_submit_v22.tar.gz /workspace/minicpm_sala_submit_v22.tar.gz && ls -la /workspace/minicpm_sala_submit_v22.tar.gz" 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && source ~/.fcloud_config && curl -s -o benchmark/soar/demo_sala/minicpm_sala_submit_v22.tar.gz "${FCLOUD_URL%/}/files/minicpm_sala_submit_v22.tar.gz?token=${FCLOUD_TOKEN}" -w "HTTP=%{http_code} size=%{size_download}\n" && sha256sum benchmark/soar/demo_sala/minicpm_sala_submit_v22.tar.gz

sha256 matches. Pause fcloud.  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -2 || (sleep 5 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -2)

Ran terminal command:  cd /home/oldzhu/sglang && sleep 8 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -2

Starting: *Document champion findings (EN+ZH)* (5/7)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Now create the champion findings doc + NVFP4/Medusa proposal. Combined into one research+proposal pair as they share the same context.  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_week7_champion_review_20260504.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_week7_champion_review_20260504.zh.md)

Now write the proposal pair (phased plan).   

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md)

Now create chat log + commit + push everything.  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_week7-champion-review_20260504_1700.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_week7-champion-review_20260504_1700.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/RESEARCH_week7_champion_review_20260504.{en,zh}.md docs/soar_2026_changes/PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.{en,zh}.md docs/soar_2026_changes/chat/CHAT_week7-champion-review_20260504_1700.{en,zh}.md && git commit -m "docs: Week 7 champion review + NVFP4/Medusa proposal (EN+ZH)

Research note + 4-phase proposal to reproduce 香草小张's Week 7 champion
recipe (score 88.35 vs team-beta 30.04):
  Phase A: NVFP4 + Marlin W4A16 baseline
  Phase B: FourOverSix adaptive block scale (M=6 vs M=4 picked by
           per-block dequant MSE, embedded inside GPTQ iteration)
  Phase C: Medusa K=1 head training (off-fcloud)
  Phase D: sglang server-side Medusa tree-verify with GLA state
           branching (per-branch fork from h_parent, no cross-branch
           pollution of recurrent state)

Each phase independently shippable behind env-gate
(SOAR_QUANT_PROFILE, SOAR_MEDUSA_ENABLE). Recommends approve A first.

Source: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
Refs: arXiv:2512.02010 (FourOverSix), Cai et al. ICML 2024 (Medusa)

Includes bilingual chat log capturing v22 packaging + leaderboard
refresh in same round." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Completed: *Document champion findings (EN+ZH)* (5/7)

## Done — v22 packaged + champion plan drafted

### v22 ready to upload

- **File**: minicpm_sala_submit_v22.tar.gz (743MB)
- **SHA256**: `d550e6ea7a1e4cedf933a76b6dcb227191abfc1e84fb42c3161e902b7cc54030`
- **Diff vs v21**: prepare_env.sh default `SOAR_TORCH_COMPILE_MAX_BS` flipped 8 → 24. Local Smax 33.62 → 32.54 (−3.2%) at acc=79.11%. Rollback: `SOAR_TORCH_COMPILE_MAX_BS=8`.

### Leaderboard refresh (2026-05-04 13:56)

| # | Team | Score | Note |
|---|------|-------|------|
| 1 | 香草小张 | **88.35** | Week 7 champ — NVFP4 FourOverSix + Medusa GLA verify |
| 2 | FlashSALA | 86.68 | Same recipe (week-on-week jump consistent) |
| 3 | dwq | 67.8 | |
| 4 | Slightwind | 62.05 | |
| 5 | 智算一队 | 50.66 | **top-5 cutoff** |
| 22 | **team-beta** | **30.04** | gap to #5: +20.62 (68% needed) |

Top-2 broke decisively from #3 (67.8) — the NVFP4+Medusa recipe is the proven lever.

### Champion review + 4-phase plan (commit `38f0ae7c0`)

- Research: RESEARCH_week7_champion_review_20260504.en.md / zh — both pillars dissected with code snippets, hardware rationale, open questions.
- Proposal: PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md / zh:

| Phase | Scope | Effort | Independently shippable? |
|-------|-------|--------|--------------------------|
| **A** | NVFP4 + Marlin W4A16 baseline (drop sparse_qkv_w8 mix) | 3-7 days | Yes → v23 if acc ≥ 78% |
| **B** | FourOverSix adaptive M=6/M=4 inside GPTQ | 1-2 days | Yes → v24 if acc gain ≥ 0.3pt |
| **C** | Train Medusa K=1 head (**needs off-fcloud H100/A100**) | 2-5 days | Yields checkpoint, packaged later |
| **D** | sglang server-side Medusa tree verify with **per-branch GLA state fork** | 5-10 days (hardest) | Yes → v25 if bitwise correctness + Smax ≤ 25s |

Each phase env-gated (`SOAR_QUANT_PROFILE`, `SOAR_MEDUSA_ENABLE`); rollback = single env unset.

### Awaiting your direction

Per repo rule (no code changes without explicit approval), three open questions:

1. **Upload v22** to https://soar.openbmb.cn manually now? (gives us the bs=24 delta vs current submission scored at 30.04)
2. **Approval scope**: `approve A` (NVFP4 baseline only) / `approve all` (commit to A-D up front) / `adjust`?
3. **Phase C hardware**: do you have an H100 or A100 box (rented or owned) available? If not, we either skip C and rely on Phase D scaffolding only, or substitute EAGLE3 (sglang-supported, no separate training of new heads needed in some configurations).

Recommendation: **upload v22 + approve A**. Phase A alone is the lowest-risk leap to validate Marlin NVFP4 is healthy on SM120; if it is, we have time within the 5/13 and 5/20 weekly windows to layer B-D.
    



