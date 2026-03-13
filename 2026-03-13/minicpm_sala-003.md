pls read https://mp.weixin.qq.com/s/mBI9osYWlBC-riR7cSt1AA and https://mp.weixin.qq.com/s/juzNNA56h1sD847qNlcPeQ which are the 1st week and 2nd week championship post and let us have discussion. pls pay special attention on KV Cache FP8 量化,权重的 4-bit 量化 and 模型结构代码优化(e.g.LayerNorm 操作可以和残差连接进行融合，这里我们可以直接复用 SGLang 现成的融合算子) . pls check my code quantaniztion gptqmodel_minicpm_sala.py to see if I am doing 权重的 4-bit 量化 and what difference could be between my and championship. in addition, when I enable kv cache fp8 as the below args:
--trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 2 --max-running-requests 30 --mem-fraction-static 0.88 --schedule-conservativeness 1.0 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --kv-cache-dtype fp8_e5m2
I would get the following exception::
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
pls review the related code to investigate why and does not that mean I have to implement fp8 tensor core support in fa2 backend or something else I missing.
