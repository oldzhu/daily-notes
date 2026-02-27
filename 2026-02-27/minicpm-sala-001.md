fcloud instance error:
...
[2026-02-27 03:34:57] Init torch distributed ends. mem usage=0.00 GB
[2026-02-27 03:34:57] MOE_RUNNER_BACKEND is not initialized, the backend will be automatically selected
[2026-02-27 03:34:57] Current Python version 3.10 is below the recommended 3.11 version. It is recommended to upgrade to Python 3.11 or higher for the best experience.
[2026-02-27 03:34:58] Ignore import error when loading sglang.srt.models.glmasr: cannot import name 'GlmAsrConfig' from 'transformers' (/app/sglang_minicpm_sala_env/lib/python3.10/site-packages/transformers/__init__.py)
[2026-02-27 03:34:59] Scheduler hit an exception: Traceback (most recent call last):
  File "/app/packages/sglang-minicpm/python/sglang/srt/managers/scheduler.py", line 2937, in run_scheduler_process
    scheduler = Scheduler(
  File "/app/packages/sglang-minicpm/python/sglang/srt/managers/scheduler.py", line 336, in __init__
    self.init_model_worker()
  File "/app/packages/sglang-minicpm/python/sglang/srt/managers/scheduler.py", line 546, in init_model_worker
    self.init_tp_model_worker()
  File "/app/packages/sglang-minicpm/python/sglang/srt/managers/scheduler.py", line 474, in init_tp_model_worker
    self.tp_worker = TpModelWorker(
  File "/app/packages/sglang-minicpm/python/sglang/srt/managers/tp_worker.py", line 240, in __init__
    self._init_model_runner()
  File "/app/packages/sglang-minicpm/python/sglang/srt/managers/tp_worker.py", line 323, in _init_model_runner
    self._model_runner = ModelRunner(
  File "/app/packages/sglang-minicpm/python/sglang/srt/model_executor/model_runner.py", line 382, in __init__
    self.initialize(min_per_gpu_memory)
  File "/app/packages/sglang-minicpm/python/sglang/srt/model_executor/model_runner.py", line 425, in initialize
    compute_initial_expert_location_metadata(
  File "/app/packages/sglang-minicpm/python/sglang/srt/eplb/expert_location.py", line 541, in compute_initial_expert_location_metadata
    return ExpertLocationMetadata.init_trivial(
  File "/app/packages/sglang-minicpm/python/sglang/srt/eplb/expert_location.py", line 92, in init_trivial
    common = ExpertLocationMetadata._init_common(server_args, model_config)
  File "/app/packages/sglang-minicpm/python/sglang/srt/eplb/expert_location.py", line 193, in _init_common
    ModelConfigForExpertLocation.from_model_config(model_config)
  File "/app/packages/sglang-minicpm/python/sglang/srt/eplb/expert_location.py", line 525, in from_model_config
    model_class, _ = get_model_architecture(model_config)
  File "/app/packages/sglang-minicpm/python/sglang/srt/model_loader/utils.py", line 105, in get_model_architecture
    architectures = resolve_transformers_arch(model_config, architectures)
  File "/app/packages/sglang-minicpm/python/sglang/srt/model_loader/utils.py", line 70, in resolve_transformers_arch
    raise ValueError(
ValueError: MiniCPMSALAForCausalLM has no SGlang implementation and the Transformers implementation is not compatible with SGLang.

[2026-02-27 03:34:59] Received sigquit from a child process. It usually means the child failed.

========

https://github.com/OpenBMB/MiniCPM
..

# Clone repository
git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
cd sglang

# One-click installation (creates venv and compiles all dependencies)
bash install_minicpm_sala.sh

# Or specify PyPI mirror
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
..

