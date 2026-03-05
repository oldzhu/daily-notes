[2026-03-04 13:10:12] Prefill batch, #new-seq: 2, #new-token: 8192, #cached-token: 0, full token usage: 0.00, mamba usage: 0.15, #running-req: 2, #queue-req: 4, 
[2026-03-04 13:10:13] Prefill batch, #new-seq: 4, #new-token: 8192, #cached-token: 0, full token usage: 0.00, mamba usage: 0.20, #running-req: 3, #queue-req: 1, 
[2026-03-04 13:10:13] Scheduler hit an exception: Traceback (most recent call last):
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
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 2288, in _forward_raw
    ret = self.forward_extend(
          ^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/model_executor/model_runner.py", line 2155, in forward_extend
    return self.model.forward(
           ^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/models/minicpm.py", line 602, in forward
    hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/models/minicpm.py", line 554, in forward
    hidden_states, residual = layer(
                              ^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/models/minicpm.py", line 489, in forward
    hidden_states = self.self_attn(
                    ^^^^^^^^^^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/models/minicpm.py", line 191, in forward
    attn_output = self.attn(q, k, v, forward_batch)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/layers/radix_attention.py", line 128, in forward
    return forward_batch.attn_backend.forward(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 1363, in forward
    return self.forward_extend(
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 1330, in forward_extend
    return self.full_attn_backend.forward_extend(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/sglang/python/sglang/srt/layers/attention/minicpm_backend.py", line 1020, in forward_extend
    metadata.sparse_page_table[sparse_page_table_idx_start, : kv_len] = page_table[dense_bs, : kv_len] * 2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: The expanded size of the tensor (3427) must match the existing size (7928) at non-singleton dimension 0.  Target sizes: [3427].  Tensor sizes: [7928]

[2026-03-04 13:10:13] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
./launch_toolkit_def001.sh: line 31:  2467 Killed                  python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT" --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --max-running-requests 20 --schedule-conservativeness 1.0 --skip-server-warmup "${extra_args[@]}"
