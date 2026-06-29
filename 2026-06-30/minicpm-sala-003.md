Approved. Let me check the console token, then start the fcloud instance. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py console-token-info 2>&1 | tail -10

Token is valid (~2 days left). Starting the instance.  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1 | tail -15

Instance starting. Let me wait for it to be reachable, then pull the latest code and build. First, give it a moment and check connectivity by pulling the repo.   

Ran terminal command:  cd /home/oldzhu/sglang && for i in 1 2 3 4 5 6; do echo "--- attempt $i ---"; python3 scripts/fcloud/fcloud_exec.py exec 'echo READY && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1' 2>&1 | tail -5 && break || sleep 20; done

Instance is up. Now pull the latest code on fcloud and build the v2 kernel.   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/sglang-minicpm && git fetch --depth 1 origin mixed_minicpm_cudagraph 2>&1 | tail -2 && git reset --hard FETCH_HEAD 2>&1 | tail -2 && git log --oneline -1' 2>&1 | tail -10

The v2 commit is on fcloud. Now check the standalone build dir exists and build with `-DUSE_V2_KERNEL=ON`.   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls -la /root/standalone_fused/ 2>&1 | head -20; echo "=== CMakeLists head ==="; head -3 /root/standalone_fused/CMakeLists.txt 2>&1' 2>&1 | tail -25

The standalone CMakeLists is stale (Jun 3, no v2 option). Let me copy the updated one from the repo into the standalone dir, then configure + build.    

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cp /root/sglang-minicpm/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt /root/standalone_fused/CMakeLists.txt && grep -n "USE_V2_KERNEL\|w4a8_fp8_qmma_v2" /root/standalone_fused/CMakeLists.txt' 2>&1 | tail -10

Now configure with v2 enabled and build (fresh build dir to avoid stale cache).    

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && rm -rf build_v2 && mkdir build_v2 && cd build_v2 && cmake .. -DTorch_DIR=/app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/share/cmake/Torch -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DUSE_V2_KERNEL=ON 2>&1 | tail -15' 2>&1 | tail -20

Configure succeeded and correctly selected the v2 kernel. Now build.     

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused/build_v2 && cmake --build . -- -j2 2>&1 | tail -30' 2>&1 | tail -35

Good progress — it compiled cleanly for `sm_120` and `sm_120a` (48 registers, no spills). The failure is a build-config issue: CMake is also emitting a default `compute_75` target, and FP8 MMA needs sm_89+. I need to tell CMake not to inject its own arch flags since we pass `-arch=sm_120a` explicitly. Let me fix the CMakeLists. 

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt#95-95), lines 95 to 115

Replacing 6 lines with 10 lines in [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt)


```

```
Now commit/push and refresh on fcloud, then rebuild.    

Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/CMakeLists_standalone.txt && git commit -q -m "standalone CMake: set CUDA_ARCHITECTURES OFF (use explicit -arch=sm_120a)

Prevents CMake from injecting a default compute_75 gencode, which fails
ptxas for FP8 mma (requires sm_89+)." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/sglang-minicpm && git fetch --depth 1 origin mixed_minicpm_cudagraph 2>&1 | tail -1 && git reset --hard FETCH_HEAD 2>&1 | tail -1 && cp /root/sglang-minicpm/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt /root/standalone_fused/CMakeLists.txt && grep -n "CUDA_ARCHITECTURES OFF" /root/standalone_fused/CMakeLists.txt' 2>&1 | tail -8

Reconfigure and rebuild (fresh, since target property changed).       

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && rm -rf build_v2 && mkdir build_v2 && cd build_v2 && cmake .. -DTorch_DIR=/app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/share/cmake/Torch -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DUSE_V2_KERNEL=ON > /tmp/cfg_v2.log 2>&1 && cmake --build . -- -j2 2>&1 | tail -20' 2>&1 | tail -25

Same error despite `CUDA_ARCHITECTURES OFF`. I need to see the actual nvcc command line to find where `compute_75` comes from. Let me inspect the verbose compile command.         

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused/build_v2 && grep -i "arch\|gencode\|sm_\|compute_" CMakeFiles/w4a8_fused_gemm.dir/build.make | head -20; echo "=== flags file ==="; cat CMakeFiles/w4a8_fused_gemm.dir/flags.make 2>/dev/null | grep -i "cuda_flags\|arch\|gencode" | head' 2>&1 | tail -30

Found it. `find_package(Torch)` injects a global gencode list including `compute_75` (which can't do FP8 MMA). The `CUDA_ARCHITECTURES OFF` property doesn't strip those because they live in `CMAKE_CUDA_FLAGS`. I'll strip the inherited `-gencode` flags after finding Torch, leaving only our explicit `-arch=sm_120a`.  

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt#42-42), lines 42 to 60

Replacing 5 lines with 11 lines in [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/CMakeLists_standalone.txt && git commit -q -m "standalone CMake: strip Torch-injected -gencode flags

find_package(Torch) adds compute_75..compute_120 gencodes globally;
compute_75 fails ptxas for FP8 mma. Strip them and rely on -arch=sm_120a." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/sglang-minicpm && git fetch --depth 1 origin mixed_minicpm_cudagraph 2>&1 | tail -1 && git reset --hard FETCH_HEAD 2>&1 | tail -1 && cp /root/sglang-minicpm/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt /root/standalone_fused/CMakeLists.txt && cd /root/standalone_fused && rm -rf build_v2 && mkdir build_v2 && cd build_v2 && cmake .. -DTorch_DIR=/app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/share/cmake/Torch -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DUSE_V2_KERNEL=ON > /tmp/cfg_v2.log 2>&1 && cmake --build . -- -j2 2>&1 | tail -15' 2>&1 | tail -20

Build succeeded — single clean `sm_120a` target, 48 registers, no spills. Now run the micro-benchmark. First locate the bench script on fcloud and check the .so.         

Starting: *Benchmark v2 vs Marlin* (7/7)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls -la /root/standalone_fused/build_v2/libw4a8_fused_gemm.so; echo "=== bench script ==="; ls -la /root/data/bench_w4a8_vs_marlin_real_weights.py 2>/dev/null; ls -la /root/sglang-minicpm/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py 2>/dev/null; echo "=== models ==="; ls /root/models/openbmb/ 2>/dev/null' 2>&1 | tail -12

The bench script on fcloud is the Jun 3 version. Let me check its full flow and whether it verifies correctness vs a reference.     

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'sed -n "120,200p" /root/data/bench_w4a8_vs_marlin_real_weights.py' 2>&1 | tail -90

The bench compares fused-vs-Marlin by speed (loads whichever `.so` is in `SOAR_W4A8_FUSED_SO`). I'll run it for v2, then v1, on a real layer shape. Let me run v2 first on `down_proj`.     

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && source /root/submission_sim/prepare_env.sh >/dev/null 2>&1; SOAR_W4A8_FUSED_SO=/root/standalone_fused/build_v2/libw4a8_fused_gemm.so /app/sglang_minicpm_sala_env/bin/python bench_w4a8_vs_marlin_real_weights.py /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 0 mlp.down_proj 2>&1 | tail -25' 2>&1 | tail -30

v2 is ~34-39 TFLOPS — essentially **the same as v1** (35-50), not better. The SMEM round-trip was apparently not the dominant bottleneck. Let me run v1 on the exact same shape for a direct apples-to-apples comparison.       

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls -la /root/standalone_fused/build/libw4a8_fused_gemm.so /root/submission_sim/libw4a8_fused_gemm.so 2>&1' 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && source /root/submission_sim/prepare_env.sh >/dev/null 2>&1; SOAR_W4A8_FUSED_SO=/root/submission_sim/libw4a8_fused_gemm.so /app/sglang_minicpm_sala_env/bin/python bench_w4a8_vs_marlin_real_weights.py /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 0 mlp.down_proj 2>&1 | grep -E "fused_w4a8|marlin|N=|Kernel" | tail -12' 2>&1 | tail -15

Direct comparison confirms the finding. Let me pause the instance immediately (cost rule), then report.        

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -6

504 is transient — retry.     

Ran terminal command:  cd /home/oldzhu/sglang && sleep 5 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -6

Instance paused (billing stopped). Let me record the result in the tracking doc.         

Completed: *Benchmark v2 vs Marlin* (7/7)

Let me record this benchmark result in the tracking doc.     

Searched for regex `W4A8|fused|TFLOPS|Marlin` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), 20 results

Let me find the micro-benchmark table where the fused W4A8 real kernel results are recorded.   

Searched for regex `micro-?bench|REAL|qmma|register-only|35-50|down_proj` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), 20 results

Let me read the micro-benchmark section to add the v2 result.    

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#258-258), lines 258 to 292

Searched for regex `Micro|micro|TFLOPS|GEMM Micro|kernel bench|vs Marlin` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), 5 results

Let me add a micro-benchmark result section at the end of the tracking doc.    

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#281-281), lines 281 to 292

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#290-290), lines 290 to 292

Replacing 2 lines with 14 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/TEST_RESULTS_TRACKING.md && git commit -q -m "Record W4A8 v2 register-only dequant micro-benchmark (Path B negative result)

v2 ~33-39 TFLOPS == v1 within noise; SMEM round-trip not the bottleneck.
Both fused kernels ~0.3-0.5x Marlin. Real limiter is lack of cp.async
pipeline + 4x redundant warp dequant." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Done. Path B is built, benchmarked, and recorded; the fcloud instance is paused.

## Result: Path B is a negative result

I implemented the register-only INT4 dequant kernel (w4a8_fp8_qmma_v2.cu), built it cleanly on SM120 (48 regs, 0 spills), and benchmarked it against v1 and Marlin on a real `down_proj` layer (N=4096, K=16384):

| Kernel | M=4096 | M=8192 | M=16384 | M=65536 |
|---|---|---|---|---|
| Marlin (W4A16) | **135.8** | **81.9** | **82.5** | **82.7** |
| v1 (FP8 SMEM round-trip) | 35.0 | 36.2 | 39.2 | 37.4 |
| **v2 (register-only dequant)** | 38.9 | 38.3 | 37.3 | 33.8 |

v2 is **within noise of v1** — the SMEM FP8 round-trip was **not** the bottleneck. Both fused kernels sit at ~0.3-0.5× Marlin.

## What this tells us

The real limiter is the **absence of a `cp.async` software pipeline** — v1/v2 both serialize weight-load → activation-load → MMA with `__syncthreads` stalls and single-buffered SMEM, so the FP8 tensor cores idle waiting on memory. Marlin's advantage is its deep async pipeline, not its dequant path. A secondary cost is the 4× redundant register dequant across the M-split warps.

So "just move dequant to registers" can't close the gap. The next lever that actually matters is proposal **Phase 2** — a full Marlin-style multi-stage `cp.async` pipeline with double-buffered SMEM. That's a larger rewrite (the proposal estimates it as the bulk of the effort).

## Where we stand

- The shipping baseline remains **v25 two-step W4A8** (81.07% accuracy, C=1.0, +8-9% speed) — untouched and still our best.
- This was a cheap experiment that ruled out a hypothesis before investing in the big pipeline rewrite.

Options from here: (1) commit to the Phase 2 async-pipeline rewrite, (2) pivot to a different catalog item, or (3) park fused W4A8. Want me to proceed with Phase 2, or stop here?

All changes committed and pushed to `minicpm-src`; instance paused (billing stopped).

