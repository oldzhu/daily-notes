  File "/root/sglang-minicpm/python/sglang/srt/layers/attention/minicpm_backend.py", line 1640, in init_forward_metadata_capture_cuda_graph
    flashinfer_wrapper.begin_forward(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/decode.py", line 1031, in plan
    self._cached_module = get_batch_prefill_module(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/prefill.py", line 374, in get_batch_prefill_module
    module = gen_batch_prefill_module(backend, *args).build_and_load()
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/jit/attention/modules.py", line 978, in gen_batch_prefill_module
    assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"
AssertionError: fp8 tensor core is not supported in fa2 backend

interesting!this error won't occur when not using gptq_marlin
