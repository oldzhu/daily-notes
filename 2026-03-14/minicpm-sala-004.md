1. 

export SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}" \
  --kv-cache-dtype fp8_e5m2 \
  --disable-cuda-graph





....

[2026-03-14 04:59:11] Decode batch, #running-req: 20, #full token: 2326796, full token usage: 0.14, mamba num: 20, mamba usage: 1.00, cuda graph: True, gen throughput (token/s): 1174.06, #queue-req: 6, 
[2026-03-14 04:59:11] Scheduler hit an exception: Traceback (most recent call last):
  File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 2974, in run_scheduler_process
    scheduler.event_loop_overlap()
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 1133, in event_loop_overlap
    batch = self.get_next_batch_to_run()
  File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 1860, in get_next_batch_to_run
    self.running_batch = self.update_running_batch(self.running_batch)
  File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 2181, in update_running_batch
    batch.prepare_for_decode()
  File "/root/submission_sim/sglang/python/sglang/srt/managers/schedule_batch.py", line 1987, in prepare_for_decode
    token_num_sparse_k1 = [(1 if seq_lens_next[batch_idx] >= kernel_size and (seq_lens_next[batch_idx] - kernel_size) % kernel_stride == 0 else 0) for batch_idx in range(bs)]
  File "/root/submission_sim/sglang/python/sglang/srt/managers/schedule_batch.py", line 1987, in <listcomp>
    token_num_sparse_k1 = [(1 if seq_lens_next[batch_idx] >= kernel_size and (seq_lens_next[batch_idx] - kernel_size) % kernel_stride == 0 else 0) for batch_idx in range(bs)]
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

