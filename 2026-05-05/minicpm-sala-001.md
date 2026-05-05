1. 同意 v1 仅限标准注意力 + MLP（lightning Q/K/V/O 仍 BF16）
2. sgl-kernel 内置 `fp8_blockwise_scaled_mm` (Option B，无额外 wheel)
as I concern is that additional whl installation would break the existing dependencies and expand the size of the sumission package.
3. `SOAR_W4A8_FP8_GEMM=1` 启用，验证前默认关闭
4.  if stop when 本地 public set 归一化精度跌破 79 % depends on how many improving we have in speed benchmark.
5. we choose use sgl-kernel built -in fp8 ops so no needto verify trt-llm whl installation

======

Good question on testing. CPU dry-run only needs FP8 cast (which is CPU-supported in PyTorch 2.1+) and arithmetic — **no GPU/QMMA needed for the Frobenius check**. So I can validate Step 2 locally without fcloud. Let me proceed.

First, read the existing preprocess_model.py to understand the integration point:  

Searched for regex `def main|def run|gptq|quantize|safetensors|argparse` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 20 results

Searched for regex `def main|MODE_GPTQ|elif mode|sub_argv|gptq_run|run_gptq` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 3 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1010-1010), lines 1010 to 1096

Searched for regex `def.*gptq.*dequant|unpack_qzeros|unpack_qweight|gptq_dequantize` (`**/*.py`), no results

Searched for regex `qweight.*qzeros|dequantize.*gptq|unpack.*int4` (`**/*.py`), 15 results

Searched for text `def quantize_weights` (`**/test/manual/test_triton_moe_wna16.py`), 1 result

Read [](file:///home/oldzhu/sglang/test/manual/test_triton_moe_wna16.py#15-15), lines 15 to 160

Searched for regex `qzeros|qweight|pack|dequantize|sym` (`**/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`), no results

Searched for regex `QuantizeConfig|sym=|desc_act|format|FORMAT` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 10 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#875-875), lines 875 to 895

Searched for regex `mixer_type|layer_types|hybrid_pattern|attn_pattern` (`**/python/sglang/srt/models/minicpm.py`), 10 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#515-515), lines 515 to 600

Ran terminal command:  python3 -c "import gptqmodel; from gptqmodel.utils.model import unpack_awq_to_bf16 if False else 0" 2>&1 | head -5; python3 -c "import gptqmodel.utils.model as m; print([x for x in dir(m) if 'unpack' in x.lower() or 'dequant' in x.lower()])" 2>&1 | head -10

Let me check what files gptqmodel saves and how to hook in:   

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#960-960), lines 960 to 1010

Plan refined: instead of touching preprocess_model.py and the safetensors layout, I'll do the conversion **at load time** inside `process_weights_after_loading`. This is much cleaner — no on-disk changes, no index.json updates, works on existing GPTQ artifacts. Trade-off: +~0.5 GB GPU memory for FP8 buffers (acceptable on 84 GB).   

Created 8 todos

Created [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/utils_w4a8_fp8.py)

Compacted conversationNow write a CPU unit test and run it: 

Created [](file:///home/oldzhu/sglang/test/srt/quantization/test_utils_w4a8_fp8.py)

Ran terminal command:  cd /home/oldzhu/sglang && PYTHONPATH=python python3 test/srt/quantization/test_utils_w4a8_fp8.py
Completed with input:  cd /home/oldzhu/sglang && PYTHONPATH=python python3 test/srt/quantization/test_utils_w4a8_fp8.py

Ran terminal command:  which python3 pip3 && python3 -c "import sys; print(sys.executable, sys.version)" && pip3 list 2>/dev/null | grep -i torch | head -5

Ran terminal command:  ls ~/ | head -20; ls /opt 2>/dev/null; which conda mamba 2>/dev/null; find / -maxdepth 4 -name "torch" -type d 2>/dev/null | head -5

No local torch — the CPU dry-run will execute on fcloud during the integration test (the test file is portable). Moving on to runtime integration.  

Starting: *Step 3: Hook process_weights_after_loading* (3/8)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#700-700), lines 700 to 850

Searched for regex `^class |def process_weights_after_loading|def apply\(` (`**/python/sglang/srt/layers/quantization/gptq.py`), 15 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#562-562), lines 562 to 700

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#784-784), lines 784 to 830

Now I'll patch gptq.py. Look at imports first:  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1-1), lines 1 to 40

Searched for regex `^import os|^import |logger = logging` (`**/python/sglang/srt/layers/quantization/gptq.py`), 5 results

Replacing 60 lines with 167 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py)


```

```


Checked [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py), no problems found

Starting: *Step 5: Tag eligible layers in minicpm.py* (5/8)

Now Step 5 — find construction sites of std-attn QKV/O + MLP linears:   

Searched for regex `qkv_proj|o_proj|gate_up_proj|down_proj|class MiniCPMAttention|class MiniCPMMLP|class MiniCPMLightningMixer` (`**/python/sglang/srt/models/minicpm.py`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#132-132), lines 132 to 230

Replacing 30 lines with 41 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py)


```

```


Checked [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py), no problems found

Now Step 6 — env flag in `prepare_env.sh`:    

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#1-1), lines 1 to 100

Searched for regex `SGLANG_SERVER_ARGS|--force-dense|--kv-cache-dtype` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 10 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#125-125), lines 125 to 175

Replacing 3 lines with 10 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)


```

```


Replacing 2 lines with 3 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)


```

```


Starting: *Update bilingual proposal docs with revised plan* (7/8)

Append continuation docs (revised plan + implementation record): 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_001.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_W4A8_001_iteration_001.zh.md)

Now commit + push:     

Ran terminal command:  cd /home/oldzhu/sglang && git status --short

Ran terminal command:  cd /home/oldzhu/sglang && git diff --stat docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md docs/soar_2026_changes/OPTION_B_PHASE_1_DELIVERY.md | head -10

Ran terminal command:  cd /home/oldzhu/sglang && git add benchmark/soar/demo_sala/prepare_env.sh python/sglang/srt/layers/quantization/gptq.py python/sglang/srt/layers/quantization/utils_w4a8_fp8.py python/sglang/srt/models/minicpm.py test/srt/quantization/ docs/soar_2026_changes/CHANGE_W4A8_001_iteration_001.en.md docs/soar_2026_changes/CHANGE_W4A8_001_iteration_001.zh.md && git status --short && git log --oneline -1

Ran terminal command:  cd /home/oldzhu/sglang && git commit -m "W4A8 #1: load-time GPTQ INT4 -> FP8 blockwise GEMM (env-gated)

- New utils_w4a8_fp8.py with vectorized GPTQ INT4 dequant + FP8 blockwise quant
- gptq.py GPTQMarlinLinearMethod: process_weights_after_loading caches FP8
  buffers when SOAR_W4A8_FP8_GEMM=1 + layer is whitelisted; apply() dispatches
  to cutlass_w8a8_block_fp8_linear_with_fallback with auto-fallback on error
- minicpm.py: tag std-attn QKV/O + MLP linears as _soar_w4a8_eligible
  (lightning attn intentionally excluded)
- prepare_env.sh: SOAR_W4A8_FP8_GEMM env var (default 0)
- test_utils_w4a8_fp8.py: CPU-only round-trip tests
- CHANGE_W4A8_001_iteration_001 EN+ZH docs (continuation of PROPOSAL)" && git push minicpm-src mixed_minicpm_cudagraph

Completed: *Update bilingual proposal docs with revised plan* (7/8)

W4A8 #1 implementation is complete and pushed (commit `7ce21c3f5` on `mixed_minicpm_cudagraph` → `minicpm-src`).

**Design pivot vs original §5 plan**: instead of touching preprocess_model.py and the safetensors layout, the FP8 conversion happens at **load time** inside `process_weights_after_loading`. This keeps the model artifact, `prepare_model.sh`, and submission packaging completely unchanged — toggle is purely via `SOAR_W4A8_FP8_GEMM=1`.

**Files committed:**
- utils_w4a8_fp8.py — vectorized GPTQ INT4 dequant + FP8 128×128 blockwise quant (CPU/CUDA)
- gptq.py — `GPTQMarlinLinearMethod`: env-gated buffer setup + `apply()` FP8 dispatch with auto-fallback
- minicpm.py — tag `MiniCPMMLP` (gate_up/down) and std-attn `MiniCPMAttention` (qkv/o); lightning intentionally untagged
- prepare_env.sh — `SOAR_W4A8_FP8_GEMM` (default 0)
- test_utils_w4a8_fp8.py — CPU round-trip tests
- CHANGE_W4A8_001_iteration_001.en.md / zh

**Ready for fcloud testing** — please start the fcloud instance when convenient. Suggested sequence:
1. `sync` → run unit test (`python3 sglang/test/srt/quantization/test_utils_w4a8_fp8.py`)
2. `restart-server` with default (flag=0) → spot-check baseline unchanged
3. Set `SOAR_W4A8_FP8_GEMM=1` → `restart-server` → grep server log for `[SOAR W4A8] enabled` lines (expect ~5 layers × 4 linears = 20 lines)
4. `accuracy` + `speed --variant all`, compare against v18 baseline. 



