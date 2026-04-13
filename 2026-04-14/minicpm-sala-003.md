Great question. This is actually a critical compatibility concern. Let me research the specific interaction between NGRAM spec decode and lightning attention (SimpleGLA recurrent layers). 

SimpleGLA spec decode compatibility

Searched for files matching `**/simple_gla_backend.py`, no matches

Searched for files matching `**/hybrid_linear_attn_backend.py`, 2 matches

Searched for files matching `**/ngram_worker.py`, 2 matches

Searched for regex `target_verify|draft_extend`, 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention)

Searched for regex `SimpleGLA|simple_gla`, 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#275-275), lines 275 to 330

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 100

Searched for regex `forward_mode|is_extend|is_decode|is_prefill|draft|target|verify` (`sglang` · `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#100-100), lines 100 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#300-300), lines 300 to 500

Searched for regex `class SimpleGLAAttnBackend|def forward` (`sglang` · `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 14 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#500-500), lines 500 to 800

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1440-1440), lines 1440 to 1700

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1700-1700), lines 1700 to 1703

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1703-1703), lines 1703 to 1703

Searched for regex `class HybridLinearAttnBackend` (`sglang` · `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 1 result

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1150-1150), lines 1150 to 1200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1228-1228), lines 1228 to 1330

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1330-1330), lines 1330 to 1360

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1360-1360), lines 1360 to 1380

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#100-100), lines 100 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/base_attn_backend.py#1-1), lines 1 to 100

Searched for regex `def forward|target_verify|draft_extend|forward_mode` (`sglang` · `python/sglang/srt/models/minicpm.py`), 7 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#450-450), lines 450 to 550

Searched for regex `NotImplementedError|raise.*Error|is_target_verify|draft_extend` (`sglang` · `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1632-1632), lines 1632 to 1680

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1210-1210), lines 1210 to 1227

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/simplegla_spec_decode_research.md)

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_vrtx_01N7ihBcaTZe4kzSyyJf7XNy__vscode-1776034399759/content.txt#1-1), lines 1 to 200

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_vrtx_01N7ihBcaTZe4kzSyyJf7XNy__vscode-1776034399759/content.txt#200-200), lines 200 to 240

This is a **critical finding**. Let me verify one more detail to be completely sure — how the actual standard attention layers are indexed in MiniCPM-SALA: 

Ran terminal command:  cd ~/sglang && grep -n "full_attn_layers\|full_attention_layer\|standard_attn\|attn_type" python/sglang/srt/models/minicpm.py | head -20

Searched for regex `full_attn|MiniCPMAttention|MiniCPMLightning|layer_type|attn_type|mixer_cls` (`**/python/sglang/srt/models/minicpm.py`), 8 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#525-525), lines 525 to 575

Good — the layer type is determined by `config.mixer_types[layer_id]`: either `"minicpm4"` (standard attention) or `"lightning"` (SimpleGLA recurrent).

---

Here's the analysis:

## NGRAM Speculative Decoding + MiniCPM Lightning Layers: **INCOMPATIBLE**

### The Core Problem

NGRAM spec decode's verification step (`target_verify`) needs to run multiple candidate tokens through the model simultaneously, then roll back to keep only accepted tokens. This fundamentally conflicts with recurrent attention:

### Why it fails with Lightning (SimpleGLA) layers:

**1. Recurrent state is sequential and irreversible**
- Lightning attention maintains a recurrent state: `state[t] = f(state[t-1], k[t], v[t])`
- During `target_verify`, N draft tokens are processed together
- If tokens 3-N are rejected, the state has already been corrupted by those tokens
- There is **no rollback mechanism** — the state stored is wrong for subsequent decode

**2. SimpleGLA has no `target_verify` guard**
- The code at hybrid_linear_attn_backend.py proceeds without checking the forward mode
- It doesn't raise `NotImplementedError` — it just does the **wrong computation silently**
- `_store_final_state()` overwrites the state with corrupted values

**3. `query_start_loc` stride breaks recurrence**
- During `target_verify`, `query_start_loc` uses `step=draft_token_num` (e.g., `[0, 5, 10, ...]`)
- `fused_recurrent_simple_gla` expects consecutive token indexing
- The stepped indexing breaks the recurrent chain

**4. 75% of layers are affected**
- 24 out of 32 layers are Lightning/SimpleGLA
- Only 8 layers use standard attention (FlashInfer, which supports spec decode fine)

### What would happen if we tried:
- Server **won't crash** (no NotImplementedError) — which is actually worse
- Lightning layers silently produce **corrupted recurrent states**
- Output quality degrades progressively as more tokens are generated
- Manifests as accuracy collapse or gibberish output

### Bottom line

**We need to remove `--speculative-algorithm NGRAM --speculative-num-draft-tokens 12` from prepare_env.sh before testing.** The bf16 RoPE + in-place residual + dense mode + FP8 KV can still be tested without NGRAM.

Want me to:
1. Remove the NGRAM flags from prepare_env.sh now?
2. Research if there's any way to make spec decode work with hybrid recurrent architectures (e.g., state checkpointing)?
