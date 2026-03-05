python3 generate_speed_datasets.py \
--prompt-source public \
--source-jsonl /root/data/perf_public_set.jsonl \
--source-prompt-field question \
--model-path /root/models/openbmb/MiniCPM-SALA \
--rows-s1 48 \
--rows-s8 72 \
--rows-smax 96 \

python3 run_soar_suite.py
--api-base http://127.0.0.1:30000
--model-path /root/models/openbmb/MiniCPM-SALA
--eval-script /root/SOAR-Toolkit/eval_model.py
--public-data /root/data/perf_public_set.jsonl
--speed-data-s1 /root/data/benchmark/soar/data_3x/speed_s1.jsonl
--speed-data-s8 /root/data/benchmark/soar/data_3x/speed_s8.jsonl
--speed-data-smax /root/data/benchmark/soar/data_3x/speed_smax.jsonl
--output-dir /root/soar_round_3x

Tip: do not pass --num-prompts here, so each tier uses its own 3x row count.


