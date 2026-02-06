Updated the main overview to be an index and split the mapping tables by pipeline stage, with EN/ZH parity and cross-links throughout. The stage-specific content now lives in separate docs and the README index reflects the new structure.

What changed

Overview converted to an index: vllm-custom-ops-mapping-table.md and vllm-custom-ops-mapping-table.zh-CN.md
New stage docs:
KV cache: vllm-custom-ops-mapping-kv-cache.md | vllm-custom-ops-mapping-kv-cache.zh-CN.md
Attention: vllm-custom-ops-mapping-attention.md | vllm-custom-ops-mapping-attention.zh-CN.md
Norm + RoPE + activation: vllm-custom-ops-mapping-norm-activation.md | vllm-custom-ops-mapping-norm-activation.zh-CN.md
Index updated: README.md and README.zh-CN.md
If you want, I can run make bilingual-check next.
