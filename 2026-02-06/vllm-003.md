Added new stage mapping tables (quant/GEMM, distributed/all-reduce, CPU backends, sampling/logits) and extended the attention + norm/RoPE tables to cover sparse indexing/top‑k and the fused QK norm+RoPE op, so the remaining custom ops now have call‑site to kernel mappings.

Details:

Expanded sparse attention/indexing coverage in vllm-custom-ops-mapping-attention.md and vllm-custom-ops-mapping-attention.zh-CN.md.
Added fused_qk_norm_rope to vllm-custom-ops-mapping-norm-activation.md and vllm-custom-ops-mapping-norm-activation.zh-CN.md.
New stage docs: vllm-custom-ops-mapping-quant-gemm.md, vllm-custom-ops-mapping-distributed-ar.md, vllm-custom-ops-mapping-cpu-backends.md, vllm-custom-ops-mapping-sampling.md plus their zh-CN counterparts.
Updated index and README links in vllm-custom-ops-mapping-table.md, vllm-custom-ops-mapping-table.zh-CN.md, README.md, and README.zh-CN.md.
Next steps (optional):

Run make bilingual-check to validate the bilingual policy.
If you want, point me at any additional ops you want mapped next.
