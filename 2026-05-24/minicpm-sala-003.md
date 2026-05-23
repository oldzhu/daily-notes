Iter-5 run-2 = same ckpt + same server config as iter-5, just a second accuracy pass (variance probe). The ckpt at `/root/models/MiniCPM-SALA-NVFP4-FOS` from iter-5 was overwritten by iter-6 — so run-2 must first **re-quantize iter-5 settings**, then run accuracy.

## Config

**Calibration (re-quant)** — iter-5 recipe:
- `SOAR_QUANT_PROFILE=nvfp4_fos`
- `SOAR_NVFP4_FOUR_OVER_SIX=1` (FOS ON)
- `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096`
- `SOAR_GPTQ_CALIBRATION_SAMPLES=32`
- `SOAR_GPTQ_CALIBRATION_SAMPLING=sequential`
- `SOAR_GPTQ_CALIBRATION_SEED=20260320`
- `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` = unset (defaults to `qa,mcq,cwe`)

**Server** — Tier1 long-ctx, modelopt_fp4 (auto-swapped via profile):
- `--quant-mode gptq` (NOT `noquant` — see lessons learned)
- `--model-path /root/models/MiniCPM-SALA-NVFP4-FOS`
- env: `SOAR_QUANT_PROFILE=nvfp4_fos`, `SOAR_NVFP4_FOUR_OVER_SIX=1`, `SOAR_TIER1_LONG_CONTEXT=1`, `SOAR_TORCH_COMPILE_MAX_BS=24`
- → resolves to: `--quantization modelopt_fp4 --attention-backend flashinfer --chunked-prefill-size 65536 --prefill-max-requests 4 --max-running-requests 24 --mem-fraction-static 0.84 --schedule-conservativeness 0.8 --force-dense-minicpm --kv-cache-dtype fp8_e5m2 --enable-fused-qk-norm-rope --enable-torch-compile --torch-compile-max-bs 24 --enable-mixed-chunk`

## Commands (when you're ready to run)

```bash
# (1) re-quant iter-5 (FOS=1)
python3 scripts/fcloud/fcloud_exec.py exec \
  'rm -rf /root/models/MiniCPM-SALA-NVFP4-FOS && \
   cd /root/submission_sim && source ./prepare_env.sh && \
   SOAR_QUANT_PROFILE=nvfp4_fos \
   SOAR_NVFP4_FOUR_OVER_SIX=1 \
   SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
   SOAR_GPTQ_CALIBRATION_SAMPLES=32 \
   SOAR_GPTQ_CALIBRATION_SAMPLING=sequential \
   SOAR_GPTQ_CALIBRATION_SEED=20260320 \
   python3 -u preprocess_model.py \
     --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS \
     --mode nvfp4'

# (2) restart server
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=1 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24
python3 scripts/fcloud/fcloud_workflow.py wait-server

# (3) accuracy run-1 (this is the "run-2 of iter-5", since iter-5 had only one run)
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS

# (4) speed bench (iter-5 didn't run speed)
python3 scripts/fcloud/fcloud_workflow.py speed --variant all

# (5) pause immediately
python3 scripts/fcloud/fcloud_workflow.py pause-instance
```

**Pass criteria**: ≥70% (abort gate). **Comparison target**: 71.24% (iter-5 run-1). Acceptable variance band ±2pt; if run-2 lands 69–73% the iter-5 result is reproducible. If run-2 < 69% or > 74%, fcloud variance is wider than assumed.

Note: re-quant is deterministic given identical seed + sampling + samples, so the *ckpt* should be byte-identical to iter-5 modulo modelopt requantize-resmooth's BF16 non-determinism. The variance we're probing is mostly server-side (scheduling, generation).

Approve and I'll start the instance.
