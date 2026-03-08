  File "/root/sglang-minicpm/python/sglang/srt/layers/attention/minicpm_backend.py", line 1777, in init_forward_metadata_replay_cuda_graph
    wrapper.begin_forward(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/decode.py", line 1045, in plan
    self._plan_info = self._cached_module.plan(
  File "python/tvm_ffi/cython/function.pxi", line 929, in tvm_ffi.core.Function.__call__
tvm.error.InternalError: Error in function 'PrefillSplitQOKVIndptr' at /root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/scheduler.cuh:522: kv_indptr[5]13138 - kv_indptr[4]17322 should be non-negative

[2026-03-08 06:06:54] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
