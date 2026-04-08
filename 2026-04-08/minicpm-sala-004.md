I maually tested the test 1 with the below sgalng starting:
python -m sglang.launch_server   --model-path /root/models/openbmb/MiniCPM-SALA-Copy   --trust-remote-code --disable-radix-cache   --attention-backend minicpm_flashinfer   --dense-as-sparse   --kv-cache-dtype fp8_e5m2   --chunked-prefill-size 32768 --max-prefill-tokens 32768   --prefill-max-requests 1 --max-running-requests 20   --mem-fraction-static 0.84 --schedule-conservativeness 1.0   --port 10000
1st time I got the below:
2026-04-07 11:16:50,541 - SGLANG_INFERENCE - ERROR - Request failed: HTTPConnectionPool(host='127.0.0.1', port=10000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=10000): Failed to establish a new connection: [Errno 111] Connection refused"))
Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [02:20<00:00,  1.07it/s]
2026-04-07 11:16:50,544 - SGLANG_INFERENCE - INFO - Processing time: 140.21s

Generation completed in 145.72 seconds

--- Evaluation Results ---

Average Score: 6.00%
Total Duration: 145.72 s
Total Tokens: In=8644166, Out=32555
Average Tokens/Sample: In=57627.8, Out=217.0
Overall TPS (Output): 223.40 tokens/s

Per-task Accuracy
  cwe: count=30 correct=0.0 accuracy=0.00% avg_in=74163.4 avg_out=9.0
  fwe: count=30 correct=0.0 accuracy=0.00% avg_in=68153.7 avg_out=9.0
  mcq: count=30 correct=8.0 accuracy=26.67% avg_in=269.9 avg_out=1034.6
  niah: count=30 correct=1.0 accuracy=3.33% avg_in=73982.7 avg_out=23.5
  qa: count=30 correct=0.0 accuracy=0.00% avg_in=71569.1 avg_out=9.0

Per-length-bucket Accuracy
  len_0_4k: count=30 correct=8.0 accuracy=26.67% avg_in=269.9 avg_out=1034.6
  len_32k_128k: count=80 correct=0.0 accuracy=0.00% avg_in=92784.4 avg_out=9.0
  len_4k_32k: count=40 correct=1.0 accuracy=2.50% avg_in=30332.8 avg_out=19.9

Per-task-length-bucket Accuracy
  task=cwe|len_32k_128k: count=20 correct=0.0 accuracy=0.00% avg_in=95471.0 avg_out=9.0
  task=cwe|len_4k_32k: count=10 correct=0.0 accuracy=0.00% avg_in=31548.2 avg_out=9.0
  task=fwe|len_32k_128k: count=20 correct=0.0 accuracy=0.00% avg_in=87381.7 avg_out=9.0
  task=fwe|len_4k_32k: count=10 correct=0.0 accuracy=0.00% avg_in=29697.8 avg_out=9.0
  task=mcq|len_0_4k: count=30 correct=8.0 accuracy=26.67% avg_in=269.9 avg_out=1034.6
  task=niah|len_32k_128k: count=20 correct=0.0 accuracy=0.00% avg_in=95307.5 avg_out=9.0
  task=niah|len_4k_32k: count=10 correct=1.0 accuracy=10.00% avg_in=31333.2 avg_out=52.6
  task=qa|len_32k_128k: count=20 correct=0.0 accuracy=0.00% avg_in=92977.5 avg_out=9.0
  task=qa|len_4k_32k: count=10 correct=0.0 accuracy=0.00% avg_in=28752.2 avg_out=9.0
  ...
  in the sglang server side, I saw the CUDA OOM and server process crashed.
  I suspect ur testing result is the same, so that I run source prepare_env.sh (as it set export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6}" which would fix the CUDA OOM) and then start the sglang using the same args above, then I got the below:
2026-04-07 12:21:23,834 - SGLANG_INFERENCE - ERROR - Request failed: HTTPConnectionPool(host='127.0.0.1', port=10000): Read timed out. (read timeout=3000)
Generating:  90%|█████████████████████████████████████████████████████████████████████████████████████████████████▏          | 135/150 [54:49<03:46, 15.08s/it]2026-04-07 12:21:37,741 - SGLANG_INFERENCE - ERROR - Request failed: HTTPConnectionPool(host='127.0.0.1', port=10000): Read timed out. (read timeout=3000)
Generating:  92%|███████████████████████████████████████████████████████████████████████████████████████████████████▎        | 138/150 [55:12<02:11, 10.92s/it]2026-04-07 12:34:03,463 - SGLANG_INFERENCE - ERROR - Request failed: HTTPConnectionPool(host='127.0.0.1', port=10000): Read timed out. (read timeout=3000)
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [1:22:02<00:00, 32.81s/it]
2026-04-07 12:48:44,215 - SGLANG_INFERENCE - INFO - Processing time: 4922.06s

Generation completed in 4927.67 seconds

--- Evaluation Results ---

Average Score: 53.18%
Total Duration: 4927.67 s
Total Tokens: In=8644166, Out=1028836
Average Tokens/Sample: In=57627.8, Out=6858.9
Overall TPS (Output): 208.79 tokens/s

Per-task Accuracy
  cwe: count=30 correct=13.1 accuracy=43.67% avg_in=74163.4 avg_out=20320.0
  fwe: count=30 correct=27.6667 accuracy=92.22% avg_in=68153.7 avg_out=5054.2
  mcq: count=30 correct=20.0 accuracy=66.67% avg_in=269.9 avg_out=7009.1
  niah: count=30 correct=11.0 accuracy=36.67% avg_in=73982.7 avg_out=1805.9
  qa: count=30 correct=8.0 accuracy=26.67% avg_in=71569.1 avg_out=105.3

Per-length-bucket Accuracy
  len_0_4k: count=30 correct=20.0 accuracy=66.67% avg_in=269.9 avg_out=7009.1
  len_32k_128k: count=80 correct=33.5333 accuracy=41.92% avg_in=92784.4 avg_out=7857.0
  len_4k_32k: count=40 correct=26.2333 accuracy=65.58% avg_in=30332.8 avg_out=4750.1

Per-task-length-bucket Accuracy
  task=cwe|len_32k_128k: count=20 correct=6.2 accuracy=31.00% avg_in=95471.0 avg_out=23644.3
  task=cwe|len_4k_32k: count=10 correct=6.9 accuracy=69.00% avg_in=31548.2 avg_out=13671.4
  task=fwe|len_32k_128k: count=20 correct=19.3333 accuracy=96.67% avg_in=87381.7 avg_out=7250.8
  task=fwe|len_4k_32k: count=10 correct=8.3333 accuracy=83.33% avg_in=29697.8 avg_out=661.0
  task=mcq|len_0_4k: count=30 correct=20.0 accuracy=66.67% avg_in=269.9 avg_out=7009.1
  task=niah|len_32k_128k: count=20 correct=6.0 accuracy=30.00% avg_in=95307.5 avg_out=421.1
  task=niah|len_4k_32k: count=10 correct=5.0 accuracy=50.00% avg_in=31333.2 avg_out=4575.5
  task=qa|len_32k_128k: count=20 correct=2.0 accuracy=10.00% avg_in=92977.5 avg_out=111.7
  task=qa|len_4k_32k: count=10 correct=6.0 accuracy=60.00% avg_in=28752.2 avg_out=92.6
  ...
  sglang server process not OOM exit this time andI suspect the the low accuracy is related with the read timeout. In fact, I remembered I seldon see timeout early, not sure if it is caused by our changes or not.
  then I want to try the test 2 withot fp8 kv cache as the below:
  python -m sglang.launch_server   --model-path /root/models/openbmb/MiniCPM-SALA-Copy   --trust-remote-code --disable-radix-cache   --attention-backend minicpm_flashinfer   --dense-as-sparse   --chunked-prefill-size 32768 --max-prefill-tokens 32768   --pre
fill-max-requests 1 --max-running-requests 20   --mem-fraction-static 0.84 --schedule-conservativeness 1.0   --port 10000
...
[2026-04-07 13:03:53] Load weight end. type=MiniCPMSALAForCausalLM, dtype=torch.bfloat16, avail mem=64.60 GB, mem usage=17.99 GB.
[2026-04-07 13:03:53] Using KV cache dtype: torch.bfloat16
[2026-04-07 13:03:53] [MiniCPMHybridReqToTokenPool] Init: size=20, max_context_len=524292, kernel_size=32, kernel_stride=16
[2026-04-07 13:03:53] Mamba Cache is allocated. max_mamba_cache_size: 20, conv_state size: 0.00GB, ssm_state size: 0.98GB 
[2026-04-07 13:03:53] KV Cache is allocated. #tokens: 6612763, K size: 25.23 GB, V size: 25.23 GB
[2026-04-07 13:03:53] Memory pool end. avail mem=12.99 GB
[2026-04-07 13:03:54] Current Python version 3.10 is below the recommended 3.11 version. It is recommended to upgrade to Python 3.11 or higher for the best experience.
[2026-04-07 13:03:54] Capture cuda graph begin. This can take up to several minutes. avail mem=12.24 GB
[2026-04-07 13:03:54] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 20]
Capturing batches (bs=20 avail_mem=11.79 GB):   0%|                                                                                      | 0/7 [00:00<?, ?it/s]
[2026-04-07 13:03:55] Scheduler hit an exception: Traceback (most recent call last):
  File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 2937, in run_scheduler_process
    scheduler = Scheduler(
  File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 336, in __init__
    self.init_model_worker()
  File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 546, in init_model_worker
    self.init_tp_model_worker()
  File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 474, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/root/submission_sim/sglang/python/sglang/srt/managers/tp_worker.py", line 240, in __init__
    self._init_model_runner()
  File "/root/submission_sim/sglang/python/sglang/srt/managers/tp_worker.py", line 323, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/root/submission_sim/sglang/python/sglang/srt/model_executor/model_runner.py", line 382, in __init__
    self.initialize(min_per_gpu_memory)
  File "/root/submission_sim/sglang/python/sglang/srt/model_executor/model_runner.py", line 568, in initialize
    self.init_device_graphs()
  File "/root/submission_sim/sglang/python/sglang/srt/model_executor/model_runner.py", line 1991, in init_device_graphs
    self.graph_runner = graph_runners[self.device](self)
  File "/root/submission_sim/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 361, in __init__
    self.capture()
  File "/root/submission_sim/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 517, in capture
    _capture_one_stream()
  File "/root/submission_sim/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 504, in _capture_one_stream
    ) = self.capture_one_batch_size(bs, forward, stream_idx)
  File "/root/submission_sim/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 723, in capture_one_batch_size
    run_once()
  File "/root/submission_sim/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 710, in run_once
    logits_output_or_pp_proxy_tensors = forward(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/root/submission_sim/sglang/python/sglang/srt/models/minicpm.py", line 749, in forward
    hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/submission_sim/sglang/python/sglang/srt/models/minicpm.py", line 698, in forward
    hidden_states, residual = layer(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/submission_sim/sglang/python/sglang/srt/models/minicpm.py", line 636, in forward
    hidden_states = self.self_attn(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/submission_sim/sglang/python/sglang/srt/models/minicpm.py", line 271, in forward
    attn_output = self.attn(q, k, v, forward_batch)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/submission_sim/sglang/python/sglang/srt/layers/radix_attention.py", line 128, in forward
    return forward_batch.attn_backend.forward(
  File "/root/submission_sim/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 1353, in forward
    return self.forward_decode(
  File "/root/submission_sim/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 1311, in forward_decode
    return self.full_attn_backend.forward_decode(
  File "/root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_backend.py", line 1187, in forward_decode
    topk_idx = self.get_topk_for_sparse(
  File "/root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_backend.py", line 738, in get_topk_for_sparse
    get_compress_k_v2(
  File "/root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py", line 220, in get_compress_k_v2
    compress_k_core_new(
  File "/root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py", line 113, in compress_k_core_new
    compress_k_complete_kernel_new[grid](
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/triton/runtime/jit.py", line 419, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/triton/runtime/jit.py", line 733, in run
    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/triton/runtime/jit.py", line 861, in _do_compile
    kernel = self.compile(src, target=target, options=options.__dict__)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/triton/compiler/compiler.py", line 300, in compile
    module = src.make_ir(target, options, codegen_fns, module_map, context)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/triton/compiler/compiler.py", line 80, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
triton.compiler.errors.CompilationError: at 174:24:
                    # PHASE 3: Perform mean pooling compression on k
                    # ====================================================================

                    # Accumulate over all tokens in this chunk
                    acc = tl.zeros([head_dim], dtype=tl.float32)

                    for token_offset in range(kernel_size):
                        # Compute k_indices for this token
                        token_y = (new_chunk_idx * kernel_stride + token_offset) + history_compress * k_stride

                        # Read k_indices from token_table
                        if token_y < token_table_cols:
                        ^
AssertionError('Mismatched type for token_k_indices between then block (int64) and else block (int32)')

[2026-04-07 13:03:55] Received sigquit from a child process. It usually means the child failed.
Killed

In addition, pls add note for fcloud auto testing in copilot instruction - run source /root/submission_sim/prepare_env.sh before start sglang to avoid cuda oom and perf_public_set.jsonl used for eval_model_001.py is under /root/data
