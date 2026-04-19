Compacted conversationLet me verify there are no conflicts between the two branches. 

Ran terminal command:  cd /home/oldzhu/sglang && git diff main...eagle3-spec-decode --name-only 2>/dev/null || echo "Branch not local, checking worktree..." && cd /home/oldzhu/sglang-eagle3 && git diff mixed_minicpm_cudagraph...eagle3-spec-decode --name-only

Ran terminal command:  cd /home/oldzhu/sglang-nvfp4 && git diff mixed_minicpm_cudagraph...nvfp4-weight-quant --name-only

**Correct — zero conflicts.** The two branches touch completely different files:

| EAGLE3 branch | NVFP4 branch |
|---|---|
| `train_eagle3_draft.py` | `quantize_nvfp4.py` |
| `minicpm3.py` (model forward hooks) | *(no model changes)* |
| `minicpm_eagle3.py` (new draft model) | *(no model changes)* |
| `fused_recurrent.py`, `hybrid_linear_attn_backend.py` (attention fixes) | *(no attention changes)* |
| CHANGE_0090 + CHANGE_0100 docs | CHANGE_0095 docs |

No overlapping files at all. They can be merged to main independently in any order without conflicts.

When fcloud is ready, shall I start with syncing and running the EAGLE3 training (takes ~30-60 min), then run NVFP4 quantization while/after that? Or would you prefer a different order?
