Compacted conversationQuick honest take before I start packaging:

**Is a Stage 2 official submission worth a slot?** My analysis:

| Aspect | Stage 2 (today) vs v22 (current submission) |
|--------|---------------------------------------------|
| Runtime behavior with `SOAR_SPEC_MEDUSA` **off** (default) | Bit-identical — MedusaWorker isn't even instantiated |
| Runtime behavior with `SOAR_SPEC_MEDUSA=1` | MedusaWorker is a pure pass-through (spec flipped to NONE before forward), so officially this is also ≈ v22 with a slightly slower boot |
| Local accuracy | 80.11% vs 79.29% (within noise) |
| Local S1/S8/Smax | within ±1% |

So submitting Stage 2 **as-is** is essentially "v22 again" — likely to score ~the same (could go either way ±2 points due to private-set variance). **Low information value, burns a slot.**

**But** there's a real reason to do it: it **de-risks Stage 3**. If we submit Stage 2 and it scores ≈ v22, we've proven on official hardware that:
1. The spec launch args (`--speculative-algorithm MEDUSA …`) are accepted by the official runtime;
2. cuda-graph capture across the 16 buckets with spec enabled doesn't OOM/crash on the official GPU;
3. The MedusaWorker pass-through doesn't break the eval harness.

Then Stage 3 official failures point cleanly at verify logic, not infrastructure.

**My recommendation**: **submit it**, but with `SOAR_SPEC_MEDUSA=1` in `prepare_env.sh` (otherwise the Medusa code path is never exercised officially, defeating the de-risk purpose). Call it **v23-medusa-passthrough**.

Before I proceed, let me confirm the current default in `prepare_env.sh`: 

Searched for text `SOAR_SPEC_MEDUSA` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 20 results

Important detail — prepare_env.sh line 404 defaults `SOAR_SPEC_MEDUSA=0`. So the **three** real options:

| Option | What official runs | Information gain |
|--------|---------------------|------------------|
| **A** — submit with default (`SOAR_SPEC_MEDUSA=0`) | Bit-identical to v22 | **Zero.** Pure waste of slot. |
| **B (recommended)** — flip default to `SOAR_SPEC_MEDUSA=1` in prepare_env.sh before packaging | MedusaWorker pass-through on official hardware | **De-risks Stage 3 infra**: confirms spec launch args, cuda-graph capture with spec enabled, and worker pass-through all work in the official environment. Expected score ≈ v22. |
| **C** — skip submission, go straight to Stage 3 | n/a | None on Stage 2; saves submission slot for a real improvement. |

My honest recommendation: **B** if you have submission slots to spare and want maximum confidence going into Stage 3; **C** if slots are scarce. **A** is strictly worse than both.

**Which do you want?** If B, I'll:
1. Edit prepare_env.sh to default `SOAR_SPEC_MEDUSA=1` (keep eager rollback opt-in).
2. Commit + push.
3. Start fcloud, sync, package per the official packaging steps (`/root/submission_sim` → `minicpm_sala_submit_v23.tar.gz`), download to demo_sala.
4. Pause fcloud, you upload to official site manually.
5. While the official eval runs, I draft the Stage 3 proposal so it's ready for your review.

If C, I jump straight to drafting the Stage 3 proposal. 


