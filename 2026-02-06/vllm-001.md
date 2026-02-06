Got it. The next deliverable will be a “Python → native implementation” mapping table with the columns you described, and for each op we’ll add:

Python call site(s) + what it does in GPT inference (mapped to the attention/KV-cache formulas and, where applicable, the relevant paper/algorithm name)
C++/CUDA/ROCm source file(s) that implement it + key comments
A short implementation summary + pseudo-code (including inputs/outputs and memory layout notes)
When you’re ready tomorrow, tell me which slice you want to start with (recommended order):

torch.ops._C_cache_ops.* (KV-cache write/reshape/swap; easiest to tie to KV-cache equations), or
torch.ops._C.paged_attention_v1/v2 (core decode attention), or
torch.ops._C_custom_ar.* (distributed all-reduce), or
a specific model path you care about (e.g., MLA, MoE, FP8).
No action needed now; we’ll pick the first namespace/op set tomorrow and I’ll generate the table as a new doc under vllm in your notes repo.
