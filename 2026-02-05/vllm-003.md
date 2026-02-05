Added an “all custom ops exposed to Python” doc and made it reproducible.

Generated catalog doc: vllm/vllm-custom-ops-catalog.md
Section 1 lists native-registered ops by torch.ops.<namespace> scanned from vLLM csrc/**
Section 2 scans vllm/**/*.py for torch.ops.* usage
Section 3 reconciles Python-used namespaces not registered by vLLM (notably torch.ops._qutlass_C and torch.ops._flashmla_extension_C)
Reproducible generator script: vllm/scripts/extract_vllm_custom_ops_catalog.py
Auto-discovers vLLM registration sites by scanning csrc/** for TORCH_LIBRARY* plus vLLM’s TORCH_LIBRARY_EXPAND(...) macro usage
Ignored generated noise in notes repo: vllm/.gitignore (filters reports and __pycache__)
If you want, next I can start the “one by one” deep dive by picking a single namespace (e.g. torch.ops._C_cache_ops or torch.ops._C attention ops) and mapping each op name → its C++/CUDA source implementation file(s).
