  File "/root/submission_sim/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 1261, in init_forward_metadata_capture_cuda_graph
    attn_backend.init_forward_metadata_capture_cuda_graph(
  File "/root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_backend.py", line 1640, in init_forward_metadata_capture_cuda_graph
    flashinfer_wrapper.begin_forward(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/decode.py", line 1031, in plan
    self._cached_module = get_batch_prefill_module(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/prefill.py", line 374, in get_batch_prefill_module
    module = gen_batch_prefill_module(backend, *args).build_and_load()
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/jit/attention/modules.py", line 978, in gen_batch_prefill_module
    assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"
AssertionError: fp8 tensor core is not supported in fa2 backend

[2026-03-13 14:24:23] Received sigquit from a child process. It usually means the child failed.
Killed
(sglang_minicpm_sala_env) root@ai-e7e98a7c52:~/submission_sim# vi /root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_backend.py
(sglang_minicpm_sala_env) root@ai-e7e98a7c52:~/submission_sim# vi /root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_
minicpm_attention_kernels.py  minicpm_backend.py            minicpm_fuse_kernel.py        minicpm_sparse_kernels.py     minicpm_sparse_utils.py       
(sglang_minicpm_sala_env) root@ai-e7e98a7c52:~/submission_sim# vi /root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py
(sglang_minicpm_sala_env) root@ai-e7e98a7c52:~/submission_sim# vi sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py
(sglang_minicpm_sala_env) root@ai-e7e98a7c52:~/submission_sim# vi sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py
(sglang_minicpm_sala_env) root@ai-e7e98a7c52:~/submission_sim# vi /root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/decode.py
(sglang_minicpm_sala_env) root@ai-e7e98a7c52:~/submission_sim# vi /root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_backend.py
(sglang_minicpm_sala_env) root@ai-e7e98a7c52:~/submission_sim# vi /root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py 
