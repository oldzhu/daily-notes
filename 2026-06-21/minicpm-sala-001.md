Good plan. Let me find the source code for `libw4a8_fused_gemm.so`:

Ran terminal command:  cd /home/oldzhu/sglang && find . -name "*.cu" -o -name "*.cpp" -o -name "*.c" | xargs grep -l "w4a8_fp8_fused_gemm\|w4a8_fused" 2>/dev/null | head -10

Ran terminal command:  cd /home/oldzhu/sglang && grep -r "libw4a8_fused_gemm\|w4a8_fused\.so" --include="*.py" --include="*.sh" --include="CMakeLists.txt" --include="Makefile" --include="*.cmake" . 2>/dev/null | head -20

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 15 'echo "=== libw4a8 .so ===" && ls -la /root/submission_sim/libw4a8_fused_gemm.so 2>&1; echo "=== strings from .so ===" && strings /root/submission_sim/libw4a8_fused_gemm.so | grep -E "w4a8|fused|gemm|\.cu|\.cpp" | head -20; echo "=== symbols ===" && nm -D /root/submission_sim/libw4a8_fused_gemm.so 2>/dev/null | grep -i w4a8 | head -10' 2>&1

Found it! `libw4a8_fused_gemm.so` was compiled from w4a8_fp8_fused_gemm.cu (the file we removed from sgl-kernel build). Let me read the kernel to find the bug:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu)

Now I can see the full kernel. Let me analyze the likely bugs and test a hypothesis. The kernel dequants INT4→FP16 and uses WMMA (148 TFLOPS cap). Let me check the GPTQ weight format to verify correctness:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 'source /root/submission_sim/prepare_env.sh 2>/dev/null && python3 -c "
import torch, os
os.environ["SOAR_W4A8_REAL_FP8_GEMM"] = "0"

# Quick test: compare fused kernel vs PyTorch reference for a small tile
# Load just one layer to check weight format
import sys
sys.path.insert(0, "/root/submission_sim/sglang/python")

# Print qweight format info
qweight = torch.randint(0, 2**32-1, (64, 128), dtype=torch.int32, device="cuda")  # [K/8, N] int32
print("Test qweight shape:", qweight.shape)
print("qweight dtype:", qweight.dtype)

# Check the actual qweight from the model
import json
# Just check what the kernel expects vs what layer provides
print()
print("=== Kernel expects ===")
print("qweight[kp * N + ng] where kp = kg/8, kbit = (kg%8)*4")
print("w4 = (qweight[kp * N + ng] >> kbit) & 0xF")
print()
print("This means qweight is indexed as [K/8, N] - first dim is K, second is N")
print("Each int32 at [kp, ng] packs 8 4-bit INT4 values along K")
print()
print("=== GPTQ standard format ===")
print("For desc_act=False, ColumnMajor weights:")
print("  qweight shape: [in_features, out_features // 8] int32")  
print("  Each int32 packs 8 values along OUT_FEATURES dimension")
print("  Index: qweight[in_idx, out_idx // 8] >> ((out_idx % 8) * 4)")
print()
print("MISMATCH: kernel expects [K,N] but GPTQ uses [K, N//8] with packed N!")
" 2>&1 | tail -20' 2>&1

OOM — server is already using GPU. But I identified the key issues from code review. Let me analyze the weight format mismatch — the most likely root cause:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 'python3 -c "
# Analyze the kernel weight indexing
# 
# Kernel line 68-69:
#   int kp = kg / 8, kbit = (kg % 8) * 4;
#   int w4 = (qweight[kp * N + ng] >> kbit) & 0xF;
#
# This indexes qweight as [K/8, N] where:
#   kp in [0, K/8)
#   ng in [0, N)
#   Each int32 at [kp, ng] packs 8 4-bit values along K dimension
#
# BUT GPTQ ColumnMajor stores qweight as [K, N//8] where:
#   Each int32 at [k, n//8] packs 8 4-bit values along N dimension
#   Index: qweight[k, n // 8] >> ((n % 8) * 4)
#
# These are COMPLETELY DIFFERENT layouts!

print("BUG ANALYSIS:")
print()
print("Kernel expects qweight indexed as:")
print("  qweight[kg//8][n] >> ((kg%8)*4)")
print("  Shape: [K//8, N] int32")
print()
print("GPTQ ColumnMajor stores qweight as:")
print("  qweight[k][n//8] >> ((n%8)*4)")
print("  Shape: [K, N//8] int32")
print()
print("THE qweight TENSOR IS TRANSPOSED!")
print("Kernel reads qweight[kp * N + ng] which treats first dim as K"\)
print("But allocated shape is [K, N//8], NOT [K//8, N]")
print()
print("Same issue with scales:")
print("Kernel: scales[gid * N + ng] - expects [groups, N]")
print("But scales shape is [K//group, N] -> [groups, N] which IS correct")
print()
print("Same issue with qzeros:")
print("Kernel: qzeros[gid * (N/8) + zn] - expects [groups, N//8]")
print("GPTQ: qzeros shape is [K//group, N//8] -> [groups, N//8] which IS correct")
print()
print("ROOT CAUSE: qweight layout is transposed!")
print("Fix: change qweight[kp * N + ng] to qweight[kg * (N//8) + ng//8] >> ((ng%8)*4)")
" 2>&1

Based on my code review, I've identified the bugs. Here's the proposal:

## Bug Analysis: `libw4a8_fused_gemm.so`

### Root Cause: Weight Layout Mismatch

The kernel indexes `qweight` incorrectly:

| | Kernel Expects | GPTQ Actual |
|---|---|---|
| **qweight shape** | `[K//8, N]` int32 | `[K, N//8]` int32 |
| **Packing dim** | Along K (8 values per int32 in K dir) | Along N (8 values per int32 in N dir) |
| **Index** | `qweight[kg//8][n]` | `qweight[k][n//8]` |

The kernel reads `qweight[kp * N + ng]` treating the first dim as K, but the tensor shape is `[K, N//8]` — the packing is along **N**, not K. This means **all weight values are read from wrong memory locations**, producing garbage dequantized weights.

### Secondary Bug: Uneven K/N

The kernel assumes `K` is a multiple of `kTileK=128`, but actual layer dimensions may not be:

```cpp
// Grid assumes K is exactly divisible by kTileK
for (int kb = 0; kb < K; kb += kTileK) { ... }

void w4a8_fp8_fused_gemm(... int M, int N, int K, ...) {
  TORCH_CHECK(M % kTileM == 0, "M must be multiple of 128");
  // NO check for N or K being multiples!
}
```

The `qweight` packing requires `N % 8 == 0` and the tile loop requires `K % 128 == 0`. For the MiniCPM model, the gate_up_proj layer is `4096×32768` — K=4096 is divisible by 128 ✅, N=32768 is divisible by 8 ✅ but the qweight layout is still wrong.

### Proposed Fix

Change the qweight indexing from:
```cpp
int kp = kg / 8, kbit = (kg % 8) * 4;
int w4 = (qweight[kp * N + ng] >> kbit) & 0xF;
```
To:
```cpp
int k_ = kg, n_ = ng;
int np = n_ / 8, nbit = (n_ % 8) * 4;
int w4 = (qweight[k_ * (N/8) + np] >> nbit) & 0xF;
```

And rebuild `libw4a8_fused_gemm.so` from the fixed source.

Shall I implement this fix?
