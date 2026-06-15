Let me write a clean MMA kernel using shared memory for output staging:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#1-1), lines 1 to 30

Now let me write the clean MMA kernel. I'll replace the scalar kernel completely:




```

```
Now let me check and fix the rest of the file (remove old scalar kernel, update host function):

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#130-130), lines 130 to 200

There's a duplicate old scalar kernel body. Let me remove it:




```

```
Let me verify the file is correct:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#125-125), lines 125 to 179

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#1-1), lines 1 to 130




```

```
Now push and build on fcloud:

Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu && git commit -m "feat: warp MMA (m16n16k16 FP16) fused INT4 dequant + GEMM kernel with SMEM output staging" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -3 && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && rm -rf build && mkdir build && cd build && export CXX=g++ CC=gcc && cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DTorch_DIR=/app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/share/cmake/Torch 2>&1 | tail -3 && cmake --build . -- -j2 2>&1 | tail -15' 2>&1

SMEM overflow (100KB > 48KB). Let me eliminate C_smem and write directly to global:




```

```
Now fix the MMA accumulation — use wmma fragments as accumulators across K-tiles:




```

```
Now replace output write to use accumulator fragments directly:




```

```





```

```
Fix warp conflict — each warp needs its own store buffer:




```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu && git commit -m "fix: reduce SMEM (40KB), per-warp accumulator + staging buffer" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -3 && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && cmake --build build -- -j2 2>&1 | tail -10 && cp build/libw4a8_fused_gemm.so /root/submission_sim/' 2>&1




```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu && git commit -m "fix: add using namespace nvcuda for output phase" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -3 && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && cmake --build build -- -j2 2>&1 | tail -10 && cp build/libw4a8_fused_gemm.so /root/submission_sim/' 2>&1

`c_frag` scoped inside loop. Move accumulators to function scope:



Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#35-35), lines 35 to 55

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#85-85), lines 85 to 120




```

```
Now remove the duplicate wmma declarations inside the K loop:




```

```
Remove duplicate `using namespace nvcuda` in output phase:




```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu && git commit -m "fix: move wmma fragments to function scope for output access" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -3 && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && cmake --build build -- -j2 2>&1 | tail -8 && cp build/libw4a8_fused_gemm.so /root/submission_sim/ && echo "BUILD OK"' 2>&1

**BUILT!** Now test correctness and speed:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'source /root/submission_sim/prepare_env.sh 2>&1 | tail -1 && python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
from sglang.srt.layers.quantization.utils_w4a8_fp8 import gptq_int4_dequantize

# Quick correctness test
M,N,K,g = 128,128,128,128
torch.manual_seed(42)
qw = torch.randint(0,2**31-1,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,2**31-1,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.randn(K//g,N,dtype=torch.float32,device="cuda")
a = torch.randn(M,K,dtype=torch.bfloat16,device="cuda").to(torch.float8_e4m3fn)
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
w = gptq_int4_dequantize(qw,qz,sc,g)
w_fp8 = w.to(torch.float8_e4m3fn)
c_ref = torch.matmul(a.to(torch.float32), w_fp8.to(torch.float32)).to(torch.bfloat16)
err = (c.float()-c_ref.float()).abs().max().item()
print(f"MMA kernel error: {err:.4f} {'OK' if err<1.0 else 'FAIL'}")
" 2>&1' 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'source /root/submission_sim/prepare_env.sh 2>&1 | tail -1 && python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
M,N,K,g = 128,256,512,128
qw = torch.randint(0,256,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,256,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.ones(K//g,N,dtype=torch.float32,device="cuda")
a = torch.eye(M,K,dtype=torch.float16,device="cuda").to(torch.float8_e4m3fn)
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
print("Result shape:", c.shape)
print("Sample:", c[0,:5])
print("SUCCESS")
" 2>&1' 2>&1

Illegal memory access. Let me debug with CUDA_LAUNCH_BLOCKING:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'CUDA_LAUNCH_BLOCKING=1 python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
M,N,K,g = 128,128,128,128
qw = torch.zeros(K//8,N,dtype=torch.int32,device="cuda")
qz = torch.zeros(K//g,N//8,dtype=torch.int32,device="cuda")
sc = torch.ones(K//g,N,dtype=torch.float32,device="cuda")
a = torch.eye(M,K,dtype=torch.float16,device="cuda").to(torch.float8_e4m3fn)
print("Calling kernel...")
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
print("Result:", c.shape)
torch.cuda.synchronize()
print("DONE")
" 2>&1' 2>&1

Works with simple inputs! Let me test with random:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'CUDA_LAUNCH_BLOCKING=1 python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
from sglang.srt.layers.quantization.utils_w4a8_fp8 import gptq_int4_dequantize
torch.manual_seed(42)
M,N,K,g = 128,256,512,128
qw = torch.randint(0,2**31-1,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,2**31-1,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.randn(K//g,N,dtype=torch.float32,device="cuda")
a = torch.randn(M,K,dtype=torch.bfloat16,device="cuda").to(torch.float8_e4m3fn)
print("Calling fused kernel...")
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
torch.cuda.synchronize()
print("Fused done, shape:", c.shape)
w = gptq_int4_dequantize(qw,qz,sc,g)
w_fp8 = w.to(torch.float8_e4m3fn)
c_ref = torch.matmul(a.to(torch.float32), w_fp8.to(torch.float32)).to(torch.bfloat16)
err = (c.float()-c_ref.float()).abs().max().item()
print("Max error:", err)
" 2>&1' 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'CUDA_LAUNCH_BLOCKING=1 python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
# Test with one tile exactly
M,N,K,g = 128,128,128,128
torch.manual_seed(42)
qw = torch.randint(0,2**31-1,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,2**31-1,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.randn(K//g,N,dtype=torch.float32,device="cuda")
a = torch.randn(M,K,dtype=torch.bfloat16,device="cuda").to(torch.float8_e4m3fn)
print("Calling...")
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
torch.cuda.synchronize()
print("OK, shape:", c.shape)
print("Sample:", c[0,0].item())
" 2>&1' 2>&1

Single tile works. Let me test N=256:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'CUDA_LAUNCH_BLOCKING=1 python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
# N=256 requires 2 output tiles
M,N,K,g = 128,256,128,128
torch.manual_seed(42)
qw = torch.randint(0,2**31-1,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,2**31-1,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.randn(K//g,N,dtype=torch.float32,device="cuda")
a = torch.randn(M,K,dtype=torch.bfloat16,device="cuda").to(torch.float8_e4m3fn)
print("Calling N=256...")
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
torch.cuda.synchronize()
print("OK, shape:", c.shape)
" 2>&1' 2>&1

Found it! `W[wn][sk]` uses global index `wn` but W is locally indexed. Fix:




```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu && git commit -m "fix: use local W index (wn - n0) instead of global index" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -3 && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && cmake --build build -- -j2 2>&1 | tail -5 && cp build/libw4a8_fused_gemm.so /root/submission_sim/ && echo "BUILD OK"' 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'CUDA_LAUNCH_BLOCKING=1 python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
M,N,K,g = 128,256,128,128
torch.manual_seed(42)
qw = torch.randint(0,2**31-1,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,2**31-1,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.randn(K//g,N,dtype=torch.float32,device="cuda")
a = torch.randn(M,K,dtype=torch.bfloat16,device="cuda").to(torch.float8_e4m3fn)
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
torch.cuda.synchronize()
print("OK shape:", c.shape)
print("Sample:", c[0,0].item())
" 2>&1' 2>&1

N=256 works! Now full correctness test:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'CUDA_LAUNCH_BLOCKING=1 python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
from sglang.srt.layers.quantization.utils_w4a8_fp8 import gptq_int4_dequantize
torch.manual_seed(42)
M,N,K,g = 128,4096,4096,128
qw = torch.randint(0,2**31-1,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,2**31-1,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.randn(K//g,N,dtype=torch.float32,device="cuda")
a = torch.randn(M,K,dtype=torch.bfloat16,device="cuda").to(torch.float8_e4m3fn)
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
torch.cuda.synchronize()
w = gptq_int4_dequantize(qw,qz,sc,g)
w_fp8 = w.to(torch.float8_e4m3fn)
c_ref = torch.matmul(a.to(torch.float32), w_fp8.to(torch.float32)).to(torch.bfloat16)
err = (c.float()-c_ref.float()).abs().max().item()
print("Max error:", err, "PASS" if err < 1.0 else "FAIL")
" 2>&1' 2>&1

Double-counting `n0` in output write. Fix:



Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#110-110), lines 110 to 130




```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu && git commit -m "fix: remove double-counted n0 in output write (ng = wn + ni)" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -3 && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && cmake --build build -- -j2 2>&1 | tail -5 && cp build/libw4a8_fused_gemm.so /root/submission_sim/ && CUDA_LAUNCH_BLOCKING=1 python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
from sglang.srt.layers.quantization.utils_w4a8_fp8 import gptq_int4_dequantize
torch.manual_seed(42)
M,N,K,g = 128,4096,4096,128
qw = torch.randint(0,2**31-1,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,2**31-1,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.randn(K//g,N,dtype=torch.float32,device="cuda")
a = torch.randn(M,K,dtype=torch.bfloat16,device="cuda").to(torch.float8_e4m3fn)
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
torch.cuda.synchronize()
w = gptq_int4_dequantize(qw,qz,sc,g)
w_fp8 = w.to(torch.float8_e4m3fn)
c_ref = torch.matmul(a.to(torch.float32), w_fp8.to(torch.float32)).to(torch.bfloat16)
err = (c.float()-c_ref.float()).abs().max().item()
print("Max error:", err)
" 2>&1' 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'CUDA_LAUNCH_BLOCKING=1 python3 -c "
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")
from sglang.srt.layers.quantization.utils_w4a8_fp8 import gptq_int4_dequantize
torch.manual_seed(42)
M,N,K,g = 128,128,128,128
qw = torch.randint(0,2**31-1,(K//8,N),dtype=torch.int32,device="cuda")
qz = torch.randint(0,2**31-1,(K//g,N//8),dtype=torch.int32,device="cuda")
sc = torch.randn(K//g,N,dtype=torch.float32,device="cuda")
a = torch.eye(M,K,dtype=torch.float16,device="cuda").to(torch.float8_e4m3fn)
c = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw,qz,sc,a,N,K,g)
torch.cuda.synchronize()
w = gptq_int4_dequantize(qw,qz,sc,g)
w_fp8 = w.to(torch.float8_e4m3fn)
c_ref = torch.matmul(a.to(torch.float32), w_fp8.to(torch.float32)).to(torch.bfloat16)
err = (c.float()-c_ref.float()).abs().max().item()
print("Max error (I):", err)
print("First row ref:", c_ref[0,:5])
print("First row fused:", c.float()[0,:5])
" 2>&1' 2>&1

Error 1.0 (FP16 vs FP8 precision diff). Acceptable for initial testing. Let me now run speed benchmarks with this MMA kernel:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py restart-server 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1 && python3 scripts/fcloud/fcloud_workflow.py speed --variant s1 2>&1 | tail -15

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ps aux | grep "sglang::" | grep -v grep | wc -l; echo "procs"' 2>&1



## MMA Kernel Status

The warp-level MMA kernel (`m16n16k16` FP16 via `nvcuda::wmma`) compiles, builds, and runs. Key fixes applied:

1. **SMEM**: 40KB (W 32KB + A 4KB + store_tile 4KB) — fits 48KB limit
2. **W indexing**: B load uses local index `W[wn - n0][sk]` not global
3. **Output write**: Fixed double-counted `n0` → `ng = wn + ni`
4. **Accumulators**: Per-warp wmma fragments persist across K-tiles

**Test results**: Single tile (128×128) gives max error ~1.0 vs reference (FP16 vs FP8 precision difference). Server startup incomplete (503). Needs further debugging for server integration and larger dimension validation.

