
tomorrow I will 
1. download https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_dataset/perf_public_set.jsonl
2. download https://github.com/OpenBMB/SOAR-Toolkit/blob/main/eval_model.py
3. run
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
...
--num_samples：（optional）最多评测样本数（调试时可以进行少样本测试）
--verbose：（optional）打印每条样本更详细的信息
4. download https://github.com/OpenBMB/SOAR-Toolkit/blob/main/bench_serving.sh and need to prepare the 3 sets of testing data?
5. download https://github.com/OpenBMB/SOAR-Toolkit/blob/main/demos/demo-sala.tar.gz 

