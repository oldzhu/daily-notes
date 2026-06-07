Compacted conversation

Searched for files matching `**/docs/soar_2026_changes/*nvfp4*fos*`, no matches

Searched for files matching `**/docs/soar_2026_changes/*four_over_six*`, 18 matches

Searched for regex `nvfp4_fos|four_over_six|FourOverSix|NVFP4_FOUR_OVER_SIX` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), 19 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#230-230), lines 230 to 247

Clarification first: there is **no "NVFP4-FOS + W4A16 Marlin" hybrid** — those are mutually exclusive loaders. NVFP4-FOS uses `--quantization modelopt_fp4` (4-bit FP4 weights via nvidia-modelopt), while W4A16 Marlin (`--quantization gptq_marlin`) is the GPTQ INT4 baseline. The `nvfp4_fos` branch in prepare_env.sh explicitly swaps Marlin for modelopt_fp4.

If you meant the **NVFP4-FOS results we have** (Phase B, CHANGE_0151), here's the full table from TEST_RESULTS_TRACKING.md:

| Test | ori_acc | norm | C | mcq | cwe | fwe | niah | qa | acc_dur | S1 / S8 / Smax | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| FOS-1 | **75.98%** | 94.97% | **0** | 63.33 | 74.33 | 92.22 | 93.33 | 56.67 | 2697s | not run | qa/cwe collapse |
| FOS-1b | 70.27% | 87.84% | 0 | 46.67 | 74.67 | 96.67 | 83.33 | 50.00 | 2537s | 175 / 47.4 / 31.0 | mcq runaway-think |
| FOS-2 | 60.73% | 75.92% | 0 | 50.00 | 60.33 | 70.00 | 73.33 | 50.00 | 3615s | 174 / 45.9 / 34.4 | conservative sched, regression |
| FOS-2b | 63.31% | 79.14% | 0 | 53.33 | 74.33 | 82.22 | 70.00 | 36.67 | 3632s | (see FOS-2) | variance |
| FOS-3 | 68.20% | 85.25% | 0 | 56.67 | 77.67 | 76.67 | 80.00 | 50.00 | 3179s | not run | Tier1 sched recovers some |
| FOS-4 | 66.00% | 82.51% | 0 | 46.67 | 66.67 | 80.00 | 93.33 | 43.33 | 3084s | not run | seqlen=4096 doesn't help |
| FOS-5 | **71.24%** | 89.05% | 0 | 53.33 | 70.67 | 92.22 | 90.00 | 50.00 | 2485s | not run | best variant (SAMPLES=32 sequential) |
| FOS-6 | 67.33% | 84.16% | 0 | 43.33 | 70.00 | 90.00 | 90.00 | 43.33 | 3164s | not run | FOS=0 worse → FOS is protective |
| **FOS-7** | **70.98%** | 88.73% | **0** | 46.67 | 76.00 | 85.56 | 100.00 | 46.67 | 2514s | **S1=173.83 / S8=46.05 / Smax=31.07** | iter-5 repro + speed |

### Headline conclusions
- **Best NVFP4-FOS run**: FOS-7 at 70.98% with speed S1=173.83s / S8=46.05s / Smax=31.07s.
- **Speed vs GPTQ Test 12 baseline** (S1=121.71 / S8=44.09 / Smax=35.86):
  - S1: **−42.8% slower** (+52s) ← decode bs=1 hurt by NVFP4 loader path
  - S8: −4.4% slower
  - Smax: **+13.4% faster** ← only tier where NVFP4 beat GPTQ
- **Score head-to-head**: GPTQ 96.0 vs NVFP4 86.7 — S1's 40% weight dominates.
- **Accuracy floor**: every NVFP4-FOS run normalized < 97% → **C=0 (eliminated)** under strict scoring.
- **Decision recorded 2026-05-09**: NVFP4-FOS parked; returned to GPTQ-FP8-dense catalog. See CHANGE_0151_phase_b_four_over_six_006.en.md and CHANGE_0151_phase_b_four_over_six_007.en.md.

### Why NVFP4-FOS is not "W4A16 Marlin"
- NVFP4-FOS: 4-bit FP4 weight storage (e2m1), BF16/FP8 activations via modelopt_fp4 cutlass kernel.
- W4A16 Marlin (current GPTQ baseline): 4-bit INT4 weight storage, BF16 activation, hand-tuned Marlin GEMM at 148 TF.
- They share the *4-bit weight bandwidth* property but use entirely different kernels and accuracy-recovery techniques (FOS scale-selection vs GPTQ Hessian-aware quant + sparse_qkv_w8 mixed-precision).

If you want a side-by-side speed/accuracy chart between **GPTQ-FP8-Dense (current v24 baseline)** and **NVFP4-FOS-7**, I can pull that together — let me know.
