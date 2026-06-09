All speed tests complete! Let me compile and pause:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'sed -i "s/SOAR_W4A8_REAL_FP8_GEMM:-1/SOAR_W4A8_REAL_FP8_GEMM:-0/" /root/submission_sim/prepare_env.sh 2>/dev/null; echo "reverted"' 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1



## 🎉 W4A8 REAL — Complete Results

### Final Scoreboard

| Metric | W4A8 REAL | v24 Baseline | Test 12 | Δ vs Test 12 |
|---|---:|---:|---:|---:|
| **Accuracy** | **81.07%** | ~77% | 79.29% | **+1.78pt** 🔥 |
| **Normalized** | ~101% | ~96% | 99.11% | **C=1.0** |
| **S1** | **110.79s** | 110.54s | 121.71s | **−9.0%** |
| **S8** | **40.51s** | 40.32s | 44.09s | **−8.1%** |
| **Smax** | **32.67s** | 33.48s | 35.86s | **−8.9%** |
| **Duration** | **2921.51s** | ~3000s | 4244s | **−31%** |

### Per-task accuracy

| Task | W4A8 REAL | Test 12 |
|---|---:|---:|
| cwe | **85.33%** | 72.00% |
| fwe | **100.00%** | 97.78% |
| mcq | 60.00% | 63.33% |
| niah | **100.00%** | 100.00% |
| qa | 60.00% | 63.33% |

### What we proved

1. **True W4A8 works** — INT4 storage + FP8 QMMA is faster than Marlin on ALL tiers
2. **Accuracy is preserved** — 81.07% is our best ever, with C=1.0
3. **The old W4A8 path (W8A8) was wrong** — our approach (INT4→FP8 on-the-fly via CUDA kernel) eliminates the weight bandwidth penalty
4. **Even without fused GEMM dequant**, the standalone CUDA dequant kernel + SM120 FP8 GEMM is 9% faster than Marlin baseline
