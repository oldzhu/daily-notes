For the current force-dense runtime, the first optimization feature will target the lightning path only, not the dense full-attention path.

Why this is the right first feature

1. Your current runtime disables sparse MiniCPM layers through prepare_env.sh, and that is enforced in model_config.py and model_config.py.
2. In that runtime, the 75% lightning layers are still active and dominate total layer count.
3. The 25% dense full-attention layers already end in the flashinfer/flashattn backend through minicpm_backend.py, minicpm_backend.py, and minicpm_backend.py, so first-pass ROI is lower there.
4. The lightning backend is concentrated in one place: hybrid_linear_attn_backend.py. That makes it the cleanest place to do a focused, reversible first feature.

What the first feature is

The feature is: 6000D-oriented lightning backend specialization for force-dense MiniCPM-SALA.

The goal is not to change model math. The goal is to reduce overhead around the lightning kernel path and make the dispatch better matched to your real workload on 6000D.

What I would change

1. Add a dedicated optimization switch inside hybrid_linear_attn_backend.py
   - I want the existing path to remain the safe fallback.
   - The new path will be guarded by a flag so rollback is instant.

2. Rework the execution policy inside `SimpleGLAAttnBackend.forward`
   - Current logic is very simple: use recurrent mode if decode or max sequence length is below 64, otherwise use chunk mode in hybrid_linear_attn_backend.py.
   - I would make that policy explicit and tunable for 6000D.
   - The likely tuning knobs are:
     - decode always recurrent
     - short extend threshold
     - medium/long extend threshold
     - optional batch-size-aware mode selection
   - The reason is that the best threshold on 6000D is unlikely to be the same as a generic threshold of 64.

3. Reduce recurrent state movement overhead
   - Right now the backend loads the temporal state with a materializing copy in hybrid_linear_attn_backend.py and writes it back in hybrid_linear_attn_backend.py.
   - I want to inspect whether we can:
     - avoid unnecessary `.contiguous()` copies
     - use a more kernel-friendly state layout
     - preallocate or reuse a staging buffer for hot paths
   - This is a realistic first-pass gain because lightning layers do this repeatedly across most layers.

4. Add lightweight profiling markers around the lightning backend boundary
   - I do not want to guess.
   - I want clear timing around:
     - state load
     - kernel call
     - state writeback
     - mode selection
   - That gives us proof whether the main cost is kernel compute or Python/tensor plumbing.

5. Keep the caller interface stable
   - I do not want to disturb the model definition unless needed.
   - The caller in minicpm.py should keep the same contract.
   - That keeps risk low and rollback simple.

What I would not do in this first feature

1. I would not touch the sparse path yet.
   - You already said it still crashes.
   - That is a separate stabilization task.

2. I would not start by rewriting the dense flashinfer attention kernel.
   - That is deeper and higher risk for a first iteration.
   - It is also only 25% of layers in your current runtime.

3. I would not change quantization, KV dtype, or serving semantics.
   - This feature is runtime-only.
   - It should preserve current correctness behavior.

Why this should help on 6000D

The reference article’s useful lesson is not “copy a 3080 kernel trick.” The useful lesson is: profile the real hot path, then reduce passes, state traffic, and bad launch choices.

For 6000D, I would apply that as follows:

1. Prefer fewer launches and less HBM traffic over overly clever small-kernel tricks.
2. Tune the recurrent/chunk boundary based on actual throughput on your hardware, not a fixed generic threshold.
3. Focus first on the 75% layer family.
4. Treat state IO as first-class cost, because lightning attention is stateful and repeated every layer.

Expected gain

1. Best case:
   - noticeable S1 improvement
   - some S8 improvement
   - modest but real S∞ improvement if lightning remains a large fraction of per-token cost

2. Most likely:
   - decode benefits first, because recurrent lightning is called every generated token across most layers
   - long-prefill may also improve if the chunk/recurrent split is currently suboptimal

3. What I do not expect:
   - a dramatic win if the external SimpleGLA kernel itself is already perfect and all overhead is inside the external package
   - in that case, this first feature still gives us the profiling evidence needed for feature 2

Main technical constraint

There is one important limitation: the actual `fused_recurrent_simple_gla` and `chunk_simple_gla` implementations are imported from the external `fla` package in hybrid_linear_attn_backend.py.

That means this first feature is intentionally scoped to the part we control first:
1. backend dispatch
2. state layout and state movement
3. execution policy
4. profiling and fallback structure

If that shows the external kernel itself is still the bottleneck, then the next feature would be either:
1. vendoring a controllable kernel path into this repo, or
2. adding an alternative in-repo lightning backend for MiniCPM-SALA

Risk level

1. Accuracy risk: low
   - math should remain the same
   - state semantics must remain exact

2. Stability risk: low to medium
   - the only sensitive part is recurrent state handling

3. Rollback risk: very low
   - I plan to keep the old path behind a fallback switch

Exact files I expect to touch after approval

1. Primary:
   - hybrid_linear_attn_backend.py

2. Possibly minor caller/config plumbing:
   - minicpm.py
   - prepare_env.sh

3. Required docs for this feature, per repo rules:
   - `docs/soar_2026_changes/CHANGE_XXXX_...en.md`
   - `docs/soar_2026_changes/CHANGE_XXXX_...zh.md`

How I will verify it

Correctness:
```bash
python3 benchmark/soar/demo_sala/eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /models/MiniCPM-SALA \
  --data_path <perf_public_set.jsonl> \
  --concurrency 32
```

Speed:
```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

If possible, before and after profiling:
```bash
nsys profile --trace=cuda,nvtx -o minicpm_lightning_profile <command>
ncu --set full --target-processes all <command>
```

Success looks like

1. No correctness drop beyond your current quantized baseline.
2. No new crashes.
3. Lower serving duration, especially in S1 and likely S8.
4. A profile showing reduced lightning backend overhead, not just noise.

Failure looks like

1. No measurable speedup.
2. Recurrent-state mismatch.
3. Improvement only on one tiny synthetic case but regression on real serving benchmarks.

If you approve this first feature, I will prepare the implementation around this exact scope:
1. `SimpleGLAAttnBackend` dispatch specialization
2. state-IO optimization
3. 6000D-tuned recurrent/chunk policy
4. fallback switch
5. EN/ZH change docs

If you want, I can next write the feature proposal in the exact repo-required format, ready for approval, with:
1. objective and expected gain
2. compliance statement
3. exact function-level edit plan
4. validation commands
5. rollback plan
