1. 
tar --exclude='__pycache__' --exclude='*.pyc' -czf minicpm_sala_submit_v1.tar.gz .
mkdir -p /root/submission_sim
tar -xzf minicpm_sala_submit_v1.tar.gz -C /root/submission_sim
cd /root/submission_sim
find . -maxdepth 2 | sort

2. 
cd /root/submission_sim
source ./prepare_env.sh

3. 
echo "$SOAR_QUANT_MODE"
echo "$SOAR_GPTQ_CALIBRATION_FILE"
echo "$SOAR_GPTQ_CALIBRATION_SAMPLES"
echo "$SGLANG_SERVER_ARGS"
python3 - <<'PY'
import importlib.metadata as m
for pkg in ["gptqmodel", "flash-attn"]:
    try:
        print(pkg, m.version(pkg))
    except Exception as e:
        print(pkg, "NOT_FOUND", e)
PY

4. 
RAW_MODEL_DIR=/root/models/openbmb/MiniCPM-SALA
OUT_DIR=/root/models/openbmb/MiniCPM-SALA-submit-sim-quant

rm -rf "$OUT_DIR"

bash ./prepare_model.sh --input "$RAW_MODEL_DIR" --output "$OUT_DIR" 2>&1 | tee /root/prepare_model_submit_sim.log

RAW_MODEL_DIR=/root/models/openbmb/MiniCPM-SALA
OUT_DIR=/root/models/openbmb/MiniCPM-SALA-submit-sim-quant

rm -rf "$OUT_DIR"

bash ./prepare_model.sh --input "$RAW_MODEL_DIR" --output "$OUT_DIR" 2>&1 | tee /root/prepare_model_submit_sim.log

5. 
python3 - <<'PY'
import json
from pathlib import Path
out_dir = Path("/root/models/openbmb/MiniCPM-SALA-submit-sim-quant")
cfg = json.loads((out_dir / "quantize_config.json").read_text())
print(cfg)
PY

6. 

MODEL_PATH=/root/models/openbmb/MiniCPM-SALA-submit-sim-quant
HOST=0.0.0.0
PORT=30000

pkill -f "sglang.launch_server" || true

read -r -a EXTRA_ARGS <<< "${SGLANG_SERVER_ARGS:-}"

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"

7. 

curl -s http://127.0.0.1:30000/health || true

8.

cd /root/data  # or wherever eval_model.py is located
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA-submit-sim-quant \
  --data_path /root/submission_sim/perf_public_set.jsonl \
  --concurrency 32 \
  --num_samples 16
