INFO  QuantizeConfig: offload_to_disk_path auto set to `./gptqmodel_offload/hqdpgrum-rkaakpxx/`                                                                                       
[preprocess] GPTQ start bits=4 group_size=128 calibration_samples=128 batch_size=2 trust_remote_code=True
INFO  Estimated Quantization BPW (bits per weight): 4.2875 bpw, based on [bits: 4, group_size: 128]                                                                                   
WARNING:fla.utils:Current Python version 3.10 is below the recommended 3.11 version. It is recommended to upgrade to Python 3.11 or higher for the best experience.                   
Traceback (most recent call last):
  File "/root/sglang/benchmark/soar/demo_sala/preprocess_model.py", line 231, in <module>
    main()
  File "/root/sglang/benchmark/soar/demo_sala/preprocess_model.py", line 212, in main
    run_gptq_quantization(
  File "/root/sglang/benchmark/soar/demo_sala/preprocess_model.py", line 133, in run_gptq_quantization
    model = GPTQModel.load(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/models/auto.py", line 355, in load
    m = cls.from_pretrained(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/models/auto.py", line 392, in from_pretrained
    return model_definition.from_pretrained(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/models/loader.py", line 244, in from_pretrained
    model = build_shell_model(cls.loader, config=config, **model_init_kwargs)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/utils/hf.py", line 161, in build_shell_model
    shell = loader.from_config(
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 453, in from_config
    return model_class._from_config(config, **kwargs)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/transformers/modeling_utils.py", line 277, in _wrapper
    return func(*args, **kwargs)
  File "/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/transformers/modeling_utils.py", line 2311, in _from_config
    model = cls(config, **kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/MiniCPM_hyphen_SALA/modeling_minicpm_sala.py", line 2866, in __init__
    self.model = MiniCPMSALAModel(config)
  File "/root/.cache/huggingface/modules/transformers_modules/MiniCPM_hyphen_SALA/modeling_minicpm_sala.py", line 2676, in __init__
    [
  File "/root/.cache/huggingface/modules/transformers_modules/MiniCPM_hyphen_SALA/modeling_minicpm_sala.py", line 2677, in <listcomp>
    MiniCPMSALADecoderLayer(config, layer_idx)
  File "/root/.cache/huggingface/modules/transformers_modules/MiniCPM_hyphen_SALA/modeling_minicpm_sala.py", line 2433, in __init__
    self.self_attn = MiniCPMInfLLMv2Attention(
  File "/root/.cache/huggingface/modules/transformers_modules/MiniCPM_hyphen_SALA/modeling_minicpm_sala.py", line 1320, in __init__
    self.config._attn_implementation == "flash_attention_2"
AssertionError: Only flash_attention_2 is supported for sparse attention

========

is flash_attentionn_2 is really required for our case, it talks long time to insatll with max_jobs ==2. consider to remove the denendency or optimize the building.

...
      Built flash-attn==2.8.3
Prepared 1 package in 47m 51s
Installed 1 package in 336ms
 + flash-attn==2.8.3
[prepare_env] SOAR_QUANT_MODE=copy
[prepare_env] SOAR_GPTQ_CALIBRATION_FILE=/root/submission_sim/perf_public_set.jsonl
[prepare_env] SOAR_GPTQ_CALIBRATION_SAMPLES=32
[prepare_env] SOAR_GPTQ_BATCH_SIZE=1
[prepare_env] SGLANG_SERVER_ARGS= --log-level info
[prepare_env] done
...
