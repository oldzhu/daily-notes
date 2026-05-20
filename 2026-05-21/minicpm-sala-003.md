Based on the session state, here are the next steps in priority order:

## Immediate next step: unblock Path C (debug silent exit)

The fastest, lowest-cost option. Current Path C (commit `e231059c8`) dies silently after modelopt's UserWarning — no traceback. Add a hard exception trap so the swallowed error surfaces:

1. Wrap the body of `run_nvfp4_quantization` in preprocess_model.py:
   ```python
   try:
       ...existing body...
   except BaseException:
       import traceback, sys
       traceback.print_exc(file=sys.stderr)
       sys.stderr.flush()
       raise
   ```
2. Also add explicit flushed `print("[trace] ...", flush=True)` markers around: model load, calib data load, `mtq.quantize` entry/exit, monkey-patch install, `export_hf_checkpoint` entry/exit.
3. Resume fcloud → re-run quantize → read the now-visible exception → fix.

## If Path C reveals an unfixable modelopt bug

Fall back to **Path E** (most reliable): skip `export_hf_checkpoint` entirely. After `mtq.quantize`, walk modules ourselves and write a hand-shaped `state_dict` with correctly-sized `(N, K/16)` fp8_e4m3 scales + `(N, K/2)` packed weights + `hf_quant_config.json`. More work, but bypasses every modelopt export quirk.

## After a working checkpoint exists

4. Verify `weight_scale.shape == (N, K/16)` with a quick safetensors inspect script.
5. `restart-server` → `wait-server` → curl smoke test.
6. `accuracy` + `speed --variant all` via `fcloud_workflow.py`.
7. Update TEST_RESULTS_TRACKING.md.
8. Write `CHANGE_0151_phase_b_four_over_six.{en,zh}.md` documenting the FOS algorithm, the three failed paths, and final results.
9. `pause-instance`.

## Recommended: ask before fcloud resume

The instance is paused (cost-saving rule). Want me to:
- **(A)** Apply the trace+except instrumentation to `preprocess_model.py`, commit, push — then ask you to confirm before `start-instance` and re-run quantize? **(recommended)**
- **(B)** Skip Path C debugging and go straight to Path E (manual export)?
- **(C)** Pause work and just review the proposal docs?
