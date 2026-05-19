B2 approved. Per copilot-instructions, fcloud usage needs explicit approval. Step 1 is a quick probe to learn modelopt 0.43's NVFP4 internal attribute names before writing the helper.  

Asked a question (fcloud_probe)

Starting: *Resume fcloud + probe modelopt 0.43 NVFP4 internals* (1/8)

Created [](file:///home/oldzhu/sglang/scripts/fcloud/probe_nvfp4_internals.py)

Ran terminal command:  cd /home/oldzhu/sglang && git add scripts/fcloud/probe_nvfp4_internals.py && git commit -m "phase-B step 1: probe modelopt 0.43 NVFP4 internals" --quiet && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -2 && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && for i in 1 2 3 4 5 6 7 8 9 10; do out=$(python3 scripts/fcloud/fcloud_exec.py exec --timeout 15 "echo READY" 2>&1 | tail -2); if echo "$out" | grep -q READY; then echo "ready at $i"; break; fi; echo "wait $i"; sleep 15; done

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 600 "cd /root/sglang-minicpm && git pull --quiet && cd /root/submission_sim && SOAR_QUANT_PROFILE=nvfp4 source ./prepare_env.sh > /tmp/prepare_env.log 2>&1; tail -5 /tmp/prepare_env.log; python3 /root/sglang-minicpm/scripts/fcloud/probe_nvfp4_internals.py 2>&1 | tail -150" 2>&1 | tail -180

Path doesn't exist. Let me find the actual model location:                   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "ls /root/models/ 2>&1; ls /root/models/openbmb/ 2>&1; find /root -maxdepth 3 -name 'MiniCPM-SALA*' -type d 2>/dev/null" 2>&1 | tail -20

The non-quantized model is `/root/models/openbmb/MiniCPM-SALA`, not `MiniCPM-SALA-Copy`. Re-run with corrected path:                    

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 900 "cd /root/submission_sim && SOAR_QUANT_PROFILE=nvfp4 source ./prepare_env.sh > /dev/null 2>&1; PROBE_SRC=/root/models/openbmb/MiniCPM-SALA python3 /root/sglang-minicpm/scripts/fcloud/probe_nvfp4_internals.py 2>&1 | tail -200" 2>&1 | tail -210

Excellent results — the probe gave us everything we need to know. Let me pause the instance first (cost rule), then summarize and design the helper:            

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && sleep 5 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -3

Paused. Key probe findings:

**modelopt 0.43 NVFP4 internals (after `mtq.quantize`, before `export_hf_checkpoint`):**

| Aspect | What we found |
|---|---|
| Module class | `QuantLinear` (single class for all quantized linears) |
| Weight storage | **Still BF16!** `target.weight` is shape (4096, 4096) bf16. modelopt at this stage does **fake-quant** — it simulates NVFP4 in forward but keeps the original BF16 weight on disk. |
| `weight_quantizer` type | `TensorQuantizer` |
| `weight_quantizer._amax` | **scalar bf16** (per-tensor, not per-block — surprise!) |
| `weight_quantizer.maxbound` | **6.0** ← this is exactly the `M` we want to flip |
| `weight_quantizer._block_sizes` | `{-1: ..., 'type': ..., 'scale_bits': ...}` — block config (block size lives at key `-1`) |
| `weight_quantizer.is_enabled` | `True` for quantized layers |
| state_dict for one Linear | `weight` (bf16), `input_quantizer._amax` (scalar), `weight_quantizer._amax` (scalar) |

**Critical insight that changes the plan:**

The per-block FP8 E4M3 scale tensor that the runtime needs **does not exist yet** at this point. It is created inside `export_hf_checkpoint`, which:
1. Reads the BF16 weight
2. Computes per-block scales using the quantizer's `maxbound` (=6.0)
3. Packs weights to uint8 NVFP4 codes
4. Writes both to safetensors

This means the cleanest implementation route is:

**Revised B2 — post-export safetensors rewrite** (still under "B2" umbrella since modelopt internals stay untouched):
1. Let modelopt export normally → `dst/` directory with `*.weight` (uint8 packed FP4) and `*.weight_scale` (FP8 E4M3) per Linear in safetensors.
2. Re-load the **original BF16 weights from the source model** (we still have them — they're the input to `prepare_model.sh`).
3. For each Linear, compute FourOverSix per-block scales and re-pack codes, then **overwrite the safetensors entries** in `dst/`.

This is even cleaner than what I wrote in the proposal because:
- We don't poke at modelopt's internal `_amax` / `maxbound` (which is per-tensor anyway, not per-block — so we couldn't have done per-block selection that way regardless)
- The output format is whatever `export_hf_checkpoint` writes (so it stays drop-in for sglang's `modelopt_fp4` loader by construction)
- We're bit-for-bit replacing two tensors per Linear; trivial to validate

**Open questions before coding (don't need fcloud — local research):**
1. Exact safetensors key names (e.g. `model.layers.X.self_attn.q_proj.weight` vs `.weight_scale` vs something else). Can verify by reading modelopt's `export_hf_checkpoint` source.
2. Block-scale tensor layout (is it `(N, K/16)` or interleaved differently — sglang's loader had `weight_scale_interleaved` as the apply-time name).
3. Is there an `input_scale` / `alpha` we should leave untouched?

Want me to:
1. **Research the modelopt export source code locally** to nail down the exact safetensors key names + scale layout, then write the helper. (No fcloud needed.)
2. **Run a second probe on fcloud** that actually calls `export_hf_checkpoint` into a tmp dir and inspects the saved files. (More definitive but burns instance time.)
3. **Skip ahead and just write the helper** based on sglang's loader code (which already declares `weight`, `weight_scale_interleaved`, `input_scale_inv`, `alpha`) and discover bugs at runtime.

Recommended: option 1.

