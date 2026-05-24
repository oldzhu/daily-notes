before we go to gptq catalog, let us dig more into on the below things:
1.  why NVFP4-FOS weigths accurray is lower than gptq INT4?
2.  why NVFP4-FOS s1/s8 speed benchmark is slow than gptq INT4 (FP8 kvcache) marlin? which looks wired seem NVFP4 tensor should have doubled TFLOPS than FP8? Code review of the different execution path  (NVFP4 and marlin kernel) may be helpful or any other profiliing or ways can help here?
answers of the two questions  not only can help to improving NVFP4+FOS but also can help on  improving on GPTQ_FP8_DENSE path if we decide to switch to it. better to have anaylsis/report on prefill/decode, compute and memory bound, Fused ops.
=======
This is a substantial analysis task. Let me dispatch parallel explorers to map the two execution paths, then write the analysis report. 

Created 6 todos

map modelopt_fp4 path

Searched codebase for "modelopt_fp4 quantization config registration", 10 results

Searched for text `modelopt_fp4`, 20 results

Searched for files matching `**/quantization_config*.py`, no matches

Searched for text `fp4_gemm cutlass_scaled_fp4`, no results

Searched for files matching `**/sgl-kernel/csrc/**/*fp4*.cu`, 6 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1068-1068), lines 1068 to 1250

Searched codebase for "fp4_gemm function entry point flashinfer cutlass", no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_entry.cu)

Searched for text `def fp4_gemm` (`**/python/**/*.py`), 2 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#1-1), lines 1 to 150

Searched for text `torch.ops.sglang fp4` (`**/python/**/*.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#120-120), lines 120 to 200

Searched for text `def cutlass_fp4_gemm` (`**/python/**/*.py`), no results

Searched for text `def flashinfer_fp4_gemm` (`**/python/**/*.py`), no results

Searched for text `cutlass_fp4_gemm flashinfer_fp4_gemm`, no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1-1), lines 1 to 300

Searched for regex `CUDA_ARCH|CMAKE_CUDA_ARCHITECTURES|sm_120|SM120` (`**/sgl-kernel/**`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#1-1), lines 1 to 150

Searched for text `ENABLE_NVFP4` (`**/sgl-kernel/**`), 16 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#200-200), lines 200 to 300

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_entry.cu)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#100-100), lines 100 to 250

Searched for text `def fp4_quantize` (`**/python/**/*.py`), no results

Searched for text `scaled_fp4_quant` (`**/sgl-kernel/**`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/python/sgl_kernel/gemm.py#170-170), lines 170 to 280

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_quant_kernels.cu#190-190), lines 190 to 220

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1068-1068), lines 1068 to 1100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#50-50), lines 50 to 55

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#200-200), lines 200 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#600-600), lines 600 to 688

Searched for text `cutlass_fp4_bf16_gemm_dispatch_sm120` (`**/sgl-kernel/csrc/gemm/**`), 3 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#688-688), lines 688 to 688

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#483-483), lines 483 to 550

Searched for regex `\.item\(\)|\.cpu\(\)|\.to_device|stream|synchronize` (`**/python/sglang/srt/layers/quantization/modelopt_quant.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1068-1068), lines 1068 to 1180

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1168-1168), lines 1168 to 1206

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#60-60), lines 60 to 100

Searched for text `group_size NVFP4 config`, no results

Read memory

Searched for text `group_size modelopt_fp4 NVFP4` (`**/docs/**`), no results

Searched for text `group_size` (`**/benchmark/soar/**`), 20 results

Searched for text `run_nvfp4_quantization SOAR_NVFP4` (`**/benchmark/soar/demo_sala/preprocess_model.py`), no results

Searched for text `nvfp4 group_size quantization` (`**/benchmark/soar/demo_sala/preprocess_model.py`), no results

Searched codebase for "NVFP4 quantization group_size configuration export", 23 results

Compacted conversation

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_011F34uZ8ge4aEiJqpDR8TL2__vscode-1778279990881/content.txt)

Now subagent B for the Marlin path. 

map gptq_marlin path

Searched for files matching `**/GPTQMarlinConfig.py`, no matches

Searched for files matching `**/gptq_marlin*.py`, no matches

Searched for files matching `**/marlin*.cu`, no matches

Searched for regex `gptq_marlin|marlin_repack`, 20 results

Searched for regex `class.*GPTQMarlin|GPTQMarlinConfig|GPTQMarlinLinearMethod`, 20 results

Searched for files matching `**/quantization/**/*.py`, 50 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#219-219), lines 219 to 380

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#563-563), lines 563 to 700

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#700-700), lines 700 to 900

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#850-850), lines 850 to 1000

Searched for regex `def apply_gptq_marlin_linear`, 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#464-464), lines 464 to 600

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin)

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#1-1), lines 1 to 1

Searched for regex `CUDA_ARCHITECTURES|SM_|80|90|120` (`**/sgl-kernel/CMakeLists.txt`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#200-200), lines 200 to 250

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#200-200), lines 200 to 400

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#400-400), lines 400 to 600

Searched for regex `mma|CUDA_ARCH|m16n8k16|TMA` (`**/sgl-kernel/csrc/gemm/marlin/marlin_template.h`), 17 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1-1), lines 1 to 100

Searched for regex `thread_config|small_batch|large_batch|sm120` (`**/sgl-kernel/csrc/gemm/marlin/kernel.h`), no results

Searched for regex `thread_config.*sm120|small_batch_thread_configs_sm120|large_batch_thread_configs_sm120`, 8 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#800-800), lines 800 to 1000

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#150-150), lines 150 to 200

Searched for regex `def gptq_int4_dequantize`, 1 result

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin_repack.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/utils_w4a8_fp8.py#1-1), lines 1 to 100

Read memory

Searched for regex `sparse_qkv_w8|_soar_w4a8_eligible|W8_LAYERS_REGEX`, 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#100-100), lines 100 to 140

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/kernel.h)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin.cuh)

Searched for regex `group_size|bits|W8_LAYERS|sparse.*attn` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 39 results

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_019Y3Rz5wa3BmDZxvGZ33pvF__vscode-1778279990927/content.txt)

I have enough material. Now I'll write the bilingual analysis report and chat log, then commit. 

Ran terminal command:  cd /home/oldzhu/sglang && date +"%Y%m%d_%H%M" && git log -1 --oneline && git rev-parse --abbrev-ref HEAD

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.zh.md)

Now the chat log: 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_perf-deepdive-fp4-vs-marlin_20260509_0703.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_perf-deepdive-fp4-vs-marlin_20260509_0703.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.en.md docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_007.zh.md docs/soar_2026_changes/chat/CHAT_perf-deepdive-fp4-vs-marlin_20260509_0703.en.md docs/soar_2026_changes/chat/CHAT_perf-deepdive-fp4-vs-marlin_20260509_0703.zh.md && git commit -m "docs(0151): perf deep-dive — NVFP4-FOS vs GPTQ Marlin (accuracy + S1/S8 gap analysis)" && git push minicpm-src mixed_minicpm_cudagraph

Pushed. Deep‑dive analysis complete.

**Q1 — accuracy gap (88.73% vs 99.11%):** Four cumulative root causes, in order of impact: (1) NVFP4 is **W4A4** (activations re‑quantized to FP4 every forward) vs Marlin **W4A16** — dominant; (2) GPTQ Hessian‑aware OBQ with inter‑column error compensation vs modelopt static max‑abs + 2‑way FOS scale search (no Hessian, no activation awareness); (3) GPTQ has `sparse_qkv_w8` (24 sensitive linears at W8) vs uniform W4 in NVFP4; (4) FP4 E2M1 non‑uniform code‑book vs uniform INT4. FOS lifts only ~3.6 pp — cannot close ~10 pp.

**Q2 — speed gap at S1/S8:** S1/S8 are memory‑/launch‑bound, not compute‑bound. NVFP4 weights are ~9% heavier (FP8 e4m3 g=16 block scales = 1 MB vs FP16 g=128 group scales = 0.25 MB per 4096² layer), the SM120 FP4 dispatcher has only one 128×M tile config (vs Marlin's 5 small‑batch configs + runtime scorer for M=1‑8), and NVFP4 pays a per‑forward `fp4_quantize` kernel each linear. FP4's 593 TFLOPS peak only materializes at large M, exactly matching the observed Smax −13.4% flip.

**Roofline + fused‑ops tables, profiling proposals (`nsys` / `ncu` / `cuobjdump`), and catalog priority guidance** are in the four committed docs (`CHANGE_0151_phase_b_four_over_six_007.{en,zh}.md` + chat log pair, commit `ac91b1afe` pushed to `minicpm-src/mixed_minicpm_cudagraph`). Net catalog implication: down‑prioritize "rewrite Marlin", up‑prioritize scheduler / KV / fused‑pre‑attn / mcq runaway‑think. 



