1.
python3 benchmark/soar/generate_speed_datasets.py \
  --profile balanced \
  --prompt-source synthetic \
  --output-dir benchmark/soar/data/balanced_shared \
  --model-path /root/models/openbmb/MiniCPM-SALA
2.
SHARED=benchmark/soar/data/balanced_shared/speed_smax.jsonl

python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 "$SHARED" \
  --speed-data-s8 "$SHARED" \
  --speed-data-smax "$SHARED" \
  --disable-tqdm
