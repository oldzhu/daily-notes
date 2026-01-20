Added it and committed.

Updated roadmap doc: build-llm-using-cpp/docs/notes/2026-01-16_project_goal_and_next_steps.md

New explicit next-step section: “Tokenizer beyond bytes (subword / BPE ‘word tokens’)”
Includes concrete scope: tokenizer+vocab, dataset switch to token IDs, vocab_size change, checkpoint compatibility, and docs/tests expectations.
Commit: 4bcb2ec “Docs: add BPE/subword tokenizer as next-step goal”

When you’re ready to implement it, tell me which tokenizer flavor you want first (classic GPT-2 BPE vs SentencePiece/BPE, and whether you want to import an existing vocab/merges or learn one from the-verdict.txt).
