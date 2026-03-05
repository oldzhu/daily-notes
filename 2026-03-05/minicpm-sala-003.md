[2026-03-05 02:24:26] Prefill batch, #new-seq: 7, #new-token: 3332, #cached-token: 0, full token usage: 0.00, mamba usage: 0.05, #running-req: 1, #queue-req: 0, 
[2026-03-05 02:24:27] Scheduler hit an exception: Traceback (most recent call last):
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 2974, in run_scheduler_process
    scheduler.event_loop_overlap()
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 1144, in event_loop_overlap
    batch_result = self.run_batch(batch)
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/scheduler.py", line 2243, in run_batch
    batch_result = self.model_worker.forward_batch_generation(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/managers/tp_worker.py", line 448, in forward_batch_generation
    out = self.model_runner.forward(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 2210, in forward
    output = self._forward_raw(
             ^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 2251, in _forward_raw
    ret = self.graph_runner.replay(
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 847, in replay
    self.replay_prepare(forward_batch, pp_proxy_tensors)
  File "/root/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 821, in replay_prepare
    attn_backend.init_forward_metadata_replay_cuda_graph(
  File "/root/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 1284, in init_forward_metadata_replay_cuda_graph
    attn_backend.init_forward_metadata_replay_cuda_graph(
  File "/root/sglang/python/sglang/srt/layers/attention/minicpm_backend.py", line 1787, in init_forward_metadata_replay_cuda_graph
    wrapper.begin_forward(
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/flashinfer/decode.py", line 1045, in plan
    self._plan_info = self._cached_module.plan(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "python/tvm_ffi/cython/function.pxi", line 923, in tvm_ffi.core.Function.__call__
tvm.error.InternalError: Error in function 'PrefillSplitQOKVIndptr' at /root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/scheduler.cuh:522: kv_indptr[5]13138 - kv_indptr[4]17322 should be non-negative

[2026-03-05 02:24:27] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
