before do NVFP4 KV proposal (P1 survey first) and/or 1-day W4-FP8 spike.  some questions about the process flow of baseline and of w8a8 on int4 bit model(fp8)
Baseline v18 (W4A16 BF16) flow:
Load: GPTQ INT4 stays packed in HBM, fp16/bf16 group scales kept. No weight inflation.
Forward: Activations BF16 (no quant). Marlin K-loop dequants INT4 → BF16 in registers, then BF16 × BF16 → FP32 MMA at 148 TF, BF16 epilogue.
KV: FP8 e5m2 in HBM, dequantized to BF16 inside FlashAttention.

W8A8-on-4-bit-model flow:
Load: dequantized INT4 → BF16 → re-quantized to FP8 e4m3 storage.
Forward: BF16 → FP8 e4m3 per-token quantize on every forward. ran a CUTLASS FP8×FP8 BF16-out GEMM (W8A8 FP8 blockwise, not INT8).
KV: FP8 e4m3 in HBM, 
why in w8a8 load time, it does  INT4 → BF16 → re-quantized to FP8, not like in the baseline which just keep int4 in hbm? what  fp16/bf16 group scales kept in baseline means?will this keeping require any register or memory resource? 
for forward, baseline v18 dequants nt4->bf16 in registers and then bf16 xBf16 ->fp32 MMA, bf16 epilogue, what does bf16 epilogue mean? convert fp32 to bf16? w8a8 does BF16 → FP8 e4m3 per-token quantize on every forward, what is the difefrence this conversion comparing with Marlin K-loop dequants INT4 → BF16 in registers? which 1 is more effccient on compute and resources and why? FP8xFP8 to BF16 no additional Fp32-> BF16, right?
for kv, w8a8 no need to convert FP8 to BF16 inside flash attention, right? as we do FP8xFP8 GEMM.
pls explain to me combined with the source code lines, functions, compute instructions, registers and memory usage so we can fully understand the difference from the code lines, code flows, calls chain, kernel ops, gpu instructsions and gpu hardware resources view, from top to bottom to understand all these comprehensively. 
=========
Created 4 todos

Trace W4A16 Marlin + W8A8 FP8 paths

Searched for files matching `**/python/sglang/srt/layers/quantization/gptq*.py`, 2 matches

Searched for files matching `**/python/sglang/srt/layers/quantization/utils_w4a8_fp8.py`, 1 match

Searched for files matching `**/sgl-kernel/csrc/gemm/marlin/*.cu`, 3 matches

Searched for files matching `**/python/sglang/srt/mem_cache/*.py`, 20 matches

Searched for files matching `**/python/sglang/srt/models/minicpm.py`, 2 matches

Searched for regex `_soar_maybe_setup_w4a8_fp8|SOAR_W4A8_FP8_GEMM` (`**/python/sglang/srt/layers/quantization/gptq.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/utils_w4a8_fp8.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#100-100), lines 100 to 250

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#250-250), lines 250 to 350

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#350-350), lines 350 to 450

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#600-600), lines 600 to 700

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#700-700), lines 700 to 850

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#800-800), lines 800 to 900

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/utils_w4a8_fp8.py#100-100), lines 100 to 150

Searched for regex `def apply` (`**/python/sglang/srt/layers/quantization/gptq.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#900-900), lines 900 to 950

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#950-950), lines 950 to 1000

Searched for files matching `**/python/sglang/srt/layers/quantization/marlin_utils.py`, 2 matches

Searched for regex `apply_gptq_marlin_linear|cutlass_w8a8_block_fp8` (`**/python/sglang/srt/layers/quantization/**`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#871-871), lines 871 to 920

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#460-460), lines 460 to 550

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_utils.py#1-1), lines 1 to 100

Searched for regex `cutlass_w8a8_block_fp8_linear_with_fallback|per_token_group_quant` (`**/python/sglang/srt/layers/quantization/fp8_utils.py`), 20 results

Searched for regex `fp8|float8|e5m2|kv_cache` (`**/python/sglang/srt/mem_cache/**`), 30 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1-1), lines 1 to 150

Searched for regex `kv_cache_dtype|e5m2|float8_e5m2` (`**/python/sglang/srt/**`), 50 results

Searched for files matching `**/python/sglang/srt/layers/attention/*.py`, 20 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_utils.py#200-200), lines 200 to 340

Searched for regex `fp8|dequant|float8` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 16 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner.py#250-250), lines 250 to 1520

Searched for regex `per_token_group_quant|scaled_fp8` (`**/python/sglang/srt/layers/quantization/fp8_kernel.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#820-820), lines 820 to 900

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_utils.py#330-330), lines 330 to 450

Searched for regex `mma.sync|cast|dequant|bf16|fp16|i4` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), 17 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#150-150), lines 150 to 250

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#300-300), lines 300 to 400

Searched for files matching `**/sgl-kernel/csrc/gemm/marlin/marlin_template.h`, 1 match

Searched for regex `kv.cache|force.dense` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 8 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_kernel.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_kernel.py#200-200), lines 200 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_kernel.py#450-450), lines 450 to 550

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01XoxYLNdkaD3c6wUptgQunG__vscode-1777330023159/content.txt)

I have all the trace data. Now writing a comprehensive Q&A document that answers each of your questions point-by-point with code references. 

Starting: *Write comprehensive explanation doc + chat log* (4/4)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.zh.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-flow-deepdive_20260428_1100.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_w4a8-flow-deepdive_20260428_1100.zh.md)

Compacted conversation

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.en.md docs/soar_2026_changes/EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.zh.md docs/soar_2026_changes/chat/CHAT_w4a8-flow-deepdive_20260428_1100.en.md docs/soar_2026_changes/chat/CHAT_w4a8-flow-deepdive_20260428_1100.zh.md && git commit -m "docs: add code-level baseline vs W8A8 FP8 flow deep-dive (R9)" && git push minicpm-src mixed_minicpm_cudagraph

Done. Wrote a comprehensive bilingual deep-dive answering all 7 of your questions with file:line references, MMA opcodes, and a step-by-step instruction timeline:

- EXPLAIN_baseline_vs_w8a8fp8_flow_20260428_1100.en.md / zh
- CHAT_w4a8-flow-deepdive / zh

**TL;DR of corrections worth highlighting:**

- **Q5 (FP8×FP8 → BF16, no FP32→BF16?)** — **Wrong.** FP8 MMA is `mma.sync...f32.e4m3.e4m3.f32`: hardware always accumulates in FP32, then epilogue casts FP32→BF16 just like the BF16 path. There's no way to skip the FP32 accumulator on Tensor Cores.
- **Q6 (FP8 KV doesn't need FP8→BF16 in attention?)** — **Wrong on two counts.** (a) The "FP8×FP8" was the linear-layer GEMM, not attention; attention is a separate kernel. (b) Our FlashAttention backend runs **BF16**, regardless of KV storage dtype, so FP8 KV is dequanted to BF16 inside attention in **both** Path A and Path B. Skipping that needs an FP8 FlashAttention kernel — we don't have one.
- **Q4 (Marlin INT4→BF16 vs W8A8 BF16→FP8 quant)** — Marlin wins by a large margin: Marlin's dequant is fused inside K-loop, in-register, overlapped with MMA, free-rides on weight memory traffic. W8A8's activation quant is a separate Triton kernel with its own kernel-launch + HBM round-trip + per-group reduce — plus W8A8 still has 2× weight HBM bandwidth penalty.

Committed as `4ca7f2a0a` and pushed to `minicpm-src/mixed_minicpm_cudagraph`. Awaiting your direction on next move (NVFP4 KV survey vs W4-FP8 CUTLASS spike). 



