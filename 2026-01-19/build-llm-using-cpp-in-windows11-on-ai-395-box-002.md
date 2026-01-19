Updated the baseline docs to cover the new fixed-offset sanity check and added copy/pastable command lines.

README update: added a new “Sanity checks (next-byte vs dataset)” section with both modes:

Random trials: --sanity-next-from-data N (example included)
Exact selection by byte offset: --sanity-offset OFF + --sanity-preview N (example included, with PowerShell --% note)
See build-llm-using-cpp/README.md
Walkthrough update: documented semantics and how it maps to gpt.forward_logits(...), plus the PowerShell tip.

See build-llm-using-cpp/docs/training_and_inference_walkthrough.md
If you tell me the exact substring you want to validate in the-verdict.txt, I can help you compute the corresponding --sanity-offset value (it’s a 0-based byte offset in the file).

Important note
With only ~60 steps on the-verdict.txt, it’s normal to get mixed results. If you want this check to become consistently correct, increase training steps (e.g. 1000–5000+) or model size.
