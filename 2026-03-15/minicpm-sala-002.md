[2026-03-15 03:28:35] Scheduler hit an exception: Traceback (most recent call last):
File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 2974, in run_scheduler_process
scheduler.event_loop_overlap()
File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
return func(*args, **kwargs)
File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 1144, in event_loop_overlap
batch_result = self.run_batch(batch)
File "/root/submission_sim/sglang/python/sglang/srt/managers/scheduler.py", line 2243, in run_batch
batch_result = self.model_worker.forward_batch_generation(
File "/root/submission_sim/sglang/python/sglang/srt/managers/tp_worker.py", line 448, in forward_batch_generation
out = self.model_runner.forward(
File "/root/submission_sim/sglang/python/sglang/srt/model_executor/model_runner.py", line 2210, in forward
output = self._forward_raw(
File "/root/submission_sim/sglang/python/sglang/srt/model_executor/model_runner.py", line 2251, in _forward_raw
ret = self.graph_runner.replay(
File "/root/submission_sim/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 858, in replay
self.graphs[graph_key].replay()
File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/cuda/graphs.py", line 141, in replay
super().replay()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. Compile with TORCH_USE_CUDA_DSA` to enable device-side assertions.

[2026-03-15 03:28:35] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
Killed
