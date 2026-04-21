Check GPTQ calibration attention mode

Searched for files matching `**/minicpm_sala.py`, no matches

Searched for files matching `**/minicpm.py`, 6 matches

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_model.sh)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 500

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py)

Searched for regex `force_dense|dense_as_sparse|attention.*mode|topk|mixer_type|sparse` (`**/python/sglang/srt/models/minicpm.py`), 20 results

Searched for regex `force_dense|dense_as_sparse|attention.*mode|topk|mixer_type|sparse` (`**/python/sglang/srt/configs/minicpm.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#500-500), lines 500 to 800

Searched for regex `force_dense|dense_as_sparse|dense|sparse|attention|topk|mixer` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#800-800), lines 800 to 1096

Searched for regex `force_dense|dense_as_sparse` (`**/python/sglang/srt/models/minicpm.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#510-510), lines 510 to 560

Searched for regex `force.dense|dense_as_sparse|force_dense_minicpm` (`**/python/sglang/**`), 20 results

**GPTQ calibration uses SPARSE attention** (the model's native mode). Here's the chain:

1. `prepare_model.sh` → `preprocess_model.py` → `GPTQModel.load()` via HuggingFace transformers
2. HuggingFace loads the model's own `modeling_minicpm_sala.py` with `trust_remote_code=True`
3. The model's native config has `mixer_types` with "minicpm4" (sparse topk) attention — that's what runs during calibration
4. **No code anywhere** in the quantization pipeline sets `force_dense` or overrides sparse→dense
5. `--force-dense-minicpm` is a sglang-only server arg — invisible during calibration

### The Mismatch

| Phase | Attention Mode |
|-------|---------------|
| **GPTQ calibration** | **Sparse** (model default, topk=64) |
| **Serving/inference** | **Dense** (`--force-dense-minicpm`) |

So the quantization error was minimized for sparse attention activation distributions, but at inference we run dense attention — different activation patterns flow through those quantized weights.

That said, since this is **W4A16** (only weights quantized, activations remain full precision), the impact is likely small. The weight values themselves don't change between sparse/dense — only the activations passing through them do, and those aren't quantized. The GPTQ calibration chooses quantization grid points that minimize output error for a given set of input activations, but for W4A16 this is a second-order effect.

**Bottom line**: There IS a sparse/dense mismatch, but it probably explains only a small fraction of any accuracy gap. It could be worth fixing eventually (G3), but it's correctly low priority. Shall I proceed to the M1 (residual scale folding) proposal?
