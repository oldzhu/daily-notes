Good point! The GPTQ model uses a mixed-precision preset. Let me check what presets are available and what the current one does. 

Searched for regex `sparse_qkv_w8|mixed_precision_preset|o_proj` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 14 results

Find GPTQ mixed-precision presets

Searched for regex `sparse_qkv_w8|mixed_precision_preset`, 20 results

Searched for files matching `**/benchmark/soar/demo_sala/**/*.{py,sh}`, 20 matches

Searched for files matching `**/prepare_model.{py,sh}`, 1 match

Searched for files matching `**/preprocess_model.py`, 1 match

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#95-95), lines 95 to 150

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_model.sh)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0066_gptq_sparse_qkv_w8.en.md)

Searched for regex `preset|sparse_qkv_w8|o_proj_w8` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 20 results

Searched for regex `mixed_precision|gptq_mixed_precision` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 9 results

Searched for regex `def.*preset|class.*preset` (`**/benchmark/soar/demo_sala/preprocess_model.py`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#440-440), lines 440 to 550

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#550-550), lines 550 to 700

Searched for regex `_resolve_sparse_layer_ids|mixer_types` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 7 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#415-415), lines 415 to 440

Searched for regex `o_proj_w8|CHANGE_0065|CHANGE_0066`, 30 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0067_gptq_sparse_qkv_w8_loader_alignment.zh.md#1-1), lines 1 to 100

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/gptq_presets_comprehensive.md)

Excellent research. You're absolutely right — the GPTQ mixed-precision preset is a critical dimension. Here's the full picture:

## Current GPTQ Model: `sparse_qkv_w8` preset

The model at `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` uses:
- **Sparse/attention layers**: Q, K, V at **W8** (higher precision for scoring)
- **All other layers**: W4 (standard GPTQ)
- **o_proj**: W4 everywhere (despite `SOAR_GPTQ_O_PROJ_BITS=8` in env — this is only used when the preset includes `o_proj_w8`)

## Available GPTQ presets

| Preset | QKV (sparse layers) | o_proj | Model size | Accuracy expectation |
|--------|---------------------|--------|------------|---------------------|
| `off` (all W4) | W4 | W4 | Smallest | Worst |
| `o_proj_w8` | W4 | W8 (all layers) | Slightly larger | Better output projection |
| **`sparse_qkv_w8`** (current) | **W8** | W4 | Slightly larger | Better sparse scoring |
| `sparse_qkv_w8_o_proj_w8` | **W8** | **W8** | Largest | Best accuracy |

## Testing all these presets requires **re-quantizing the model** each time (~75 min per run). So the dimension you're raising adds this:

| Model variant | Already tested? | Notes |
|---|---|---|
| GPTQ `sparse_qkv_w8` | Yes (Tests 5-10) | Current model |
| GPTQ `sparse_qkv_w8_o_proj_w8` | **No** | Could improve accuracy further |
| GPTQ `o_proj_w8` | **No** | Different tradeoff |
| GPTQ `off` (all W4) | **No** | Worst accuracy baseline |

However, re-quantizing is expensive (75 min + model prep). For the **immediate priority of comparing runtime configs** (FP8/bf16 KV × sparse/dense), we can test all those with the **current GPTQ model** without re-quantization.

### Recommended testing plan

**Phase 1 — Runtime config matrix (no re-quantization needed, ~40 min each):**
1. Non-quant + FP8 + sparse (re-test clean, no 0070 bug)  
2. Non-quant + FP8 + dense  
3. Non-quant + bf16 + dense  
4. GPTQ + bf16 + dense  

**Phase 2 — GPTQ preset matrix (requires re-quantization, ~75 min prep + 40 min test each):**
5. GPTQ `sparse_qkv_w8_o_proj_w8` + bf16 KV + sparse  
6. GPTQ `sparse_qkv_w8_o_proj_w8` + FP8 KV + dense  

The combo preset (`sparse_qkv_w8_o_proj_w8`) is the most promising because it keeps W8 for both the scoring-critical QKV and the output projection, maximizing accuracy while still using GPTQ for speed.

Want me to start with Phase 1 (needs fcloud started), or do you want to prioritize the re-quantization experiments in Phase 2?
