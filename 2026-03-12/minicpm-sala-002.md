1.
python3 generate_speed_datasets.py \
  --profile balanced \
  --prompt-source synthetic \
  --output-dir benchmark/soar/data/balanced_shared_v1 \
  --model-path /root/models/openbmb/MiniCPM-SALA-submit-sim-quant

2. 
SHARED=benchmark/soar/data/balanced_shared_v1/speed_smax.jsonl

python3 run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA-submit-sim-quant \
  --speed-data-s1 "$SHARED" \
  --speed-data-s8 "$SHARED" \
  --speed-data-smax "$SHARED" \
  --disable-tqdm

3.
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA-submit-sim-quant \
  --data_path perf_public_set.jsonl \
  --concurrency 32 

4. 
nvidia-smi dmon -s pucvmet -d 1 -o TD -f benchmark/soar/results/nvidia_dmon.log

5.
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --delay 20 \
  --duration 90 \
  -o benchmark/soar/results/nsys_minicpm_probe \

MODEL_PATH=/root/models/openbmb/MiniCPM-SALA-submit-sim-quant
HOST=0.0.0.0
PORT=30000
pkill -f "sglang.launch_server" || true
read -r -a EXTRA_ARGS <<< "${SGLANG_SERVER_ARGS:-}"
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA_ARGMODEL_PATH=/root/models/openbmb/MiniCPM-SALA-submit-sim-quant
HOST=0.0.0.0
PORT=30000

pkill -f "sglang.launch_server" || true

read -r -a EXTRA_ARGS <<< "${SGLANG_SERVER_ARGS:-}"

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"S[@]}"
  
  -enable-metrics \
  --export-metrics-to-file \
  --export-metrics-to-file-dir benchmark/soar/results/request_metrics
6. 
perf record -F 99 -g -p <SERVER_PID> -- sleep 60
perf report
