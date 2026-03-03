1. modify minicpm_backend.py at line 448
...
if seqlen_q_sparse_tensor.numel() == 0:
    metadata.max_seqlen_q_adjusted = 0
...
2. CUDA oom when running eval

seems works after modify the start_sglang.sh

source /root/sglang/sglang_minicpm_sala_env/bin/activate
export MODEL_PATH=/root/models/openbmb/MiniCPM-SALA/
python3 -m sglang.launch_server  \
--model-path /root/models/openbmb/MiniCPM-SALA \
--host 0.0.0.0   \
--port 30000   \
--trust-remote-code \
--disable-radix-cache \
--attention-backend minicpm_flashinfer \
--chunked-prefill-size 4096 \
--max-running-requests 12 \
--prefill-max-requests 1 \
--max-prefill-tokens 8192 \
--mem-fraction-static 0.80 \
--schedule-conservativeness 1.2 \
--skip-server-warmup \
#--dense-as-sparse

and modify run_eval.sh to decrease concurrency from 32 to 8(16 still oom)

export MODEL_DIR=/root/models/openbmb/MiniCPM-SALA/
export DATA_DIR=/root/data
python3 ${DATA_DIR}/eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path ${MODEL_DIR} \
  --data_path ${DATA_DIR}/perf_public_set.jsonl \
  --concurrency 8


...
[2026-03-03 11:39:17] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.14, mamba usage: 0.67, #running-req: 7, #queue-req: 0, 
[2026-03-03 11:39:17] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.15, mamba usage: 0.67, #running-req: 7, #queue-req: 0, 
[2026-03-03 11:39:18] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.15, mamba usage: 0.67, #running-req: 7, #queue-req: 0, 
[2026-03-03 11:39:19] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.15, mamba usage: 0.67, #running-req: 7, #queue-req: 0, 
[2026-03-03 11:39:20] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.15, mamba usage: 0.67, #running-req: 7, #queue-req: 0, 
[2026-03-03 11:39:20] Prefill batch, #new-seq: 1, #new-token: 2026, #cached-token: 0, full token usage: 0.15, mamba usage: 0.67, #running-req: 7, #queue-req: 0, 
[2026-03-03 11:39:21] Decode batch, #running-req: 8, #full token: 924599, full token usage: 0.15, mamba num: 8, mamba usage: 0.67, cuda graph: True, gen throughput (token/s): 26.46, #queue-req: 0, 
[2026-03-03 11:39:22] Decode batch, #running-req: 8, #full token: 924943, full token usage: 0.15, mamba num: 8, mamba usage: 0.67, cuda graph: True, gen throughput (token/s): 347.72, #queue-req: 0, 
[2026-03-03 11:39:23] Decode batch, #running-req: 8, #full token: 925292, full token usage: 0.15, mamba num: 8, mamba usage: 0.67, cuda graph: True, gen throughput (token/s): 347.21, #queue-req: 0, 
... 
