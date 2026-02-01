Synced backend docs to current reality: backend seam exists (KernelBackend + registry; matmul/bmm dispatch), and fixed src/backends/* → src/backend/* in docs/backend_design_cpu_cuda_hip.md, docs/backend_design_cpu_cuda_hip.zh-CN.md, docs/backend_implementation_plan.md, docs/backend_implementation_plan.zh-CN.md.
Synced tokenizer/training docs: tokenization is pluggable, but the current training + dataset sanity checks are byte-dataset-specific (ByteDataset, 
𝑉
=
256
V=256) in docs/training_and_inference_walkthrough.md, docs/training_and_inference_walkthrough.zh-CN.md, data/README.md, data/README.zh-CN.md.
Updated “math→code”/autograd notes to avoid “always byte-level / matmul is hardcoded loops” claims: docs/transformer_math_to_code.md, docs/transformer_math_to_code.zh-CN.md, docs/autograd_and_memory_layout.md, docs/autograd_and_memory_layout.zh-CN.md.
Updated the tokenizer decision note with a “status update” reflecting the existing Tokenizer interface + ByteTokenizer + minimal BPE tokenizer: docs/tokenizer_choice_gpt2_bpe_vs_sentencepiece.md, docs/tokenizer_choice_gpt2_bpe_vs_sentencepiece.zh-CN.md.
Fixed CLI behavior to match docs and avoid confusing runs in src/main.cpp: generation prints Tokenizer::decode(...) output for non-byte tokenizers; --ascii-only/--escape-bytes are byte-tokenizer-only; training/sanity now require --tokenizer byte until a token-id dataset path exists.
Added today’s bilingual daily note: docs/notes/2026-02-01_doc_sync_audit.md, docs/notes/2026-02-01_doc_sync_audit.zh-CN.md. Verified with ctest -C Debug (pass).
Want me to git add/commit these doc+CLI sync changes (and if so, what commit message prefix do you prefer)?
