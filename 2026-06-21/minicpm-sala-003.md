Server with fused kernel timed out. Let me check the logs:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 40 2>&1 | tail -45

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 'cat /root/submission_sim/server_output_2.log 2>/dev/null | tail -30 || echo "No terminal-2 log found"' 2>&1

The server crashed because the fused kernel requires `M % 128 == 0` but CUDA graphs use small batch sizes (1-24). Let me pause and summarize:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && sleep 5 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1

Completed: *Build successful - .so created* (3/5)

## Session Summary

### Key Discovery: SM120 ≠ SM100

The competition GPU (SM120 RTX PRO) has a **different instruction set** from SM100 data center GPUs:

| | SM100 (dc) | SM120 (RTX PRO) |
|---|---|---|
| `tcgen05` QMMA | ✅ | ❌ Not supported |
| `elect_one.sync` | ✅ | ❌ |
| `cta_group::1` warpgroup | ✅ | ❌ |
| **FP8 `mma.sync`** | ✅ | ✅ **Supported!** |
| FP8 throughput | 296 TFLOPS | **296 TFLOPS** |

### What Was Built

- **w4a8_fp8_qmma.cu**: 176-line kernel using warp-level `mma.sync.aligned.m16n8k32` with FP8 e4m3 types
- **Compiles and links**: 1.2MB `libw4a8_fused_gemm.so` built at 09:45
- **SMEM**: 16KB (vs 36KB for old WMMA kernel)
- **Target**: 296 TFLOPS via SM120 FP8 tensor cores

### Blocker: Tile Size

The kernel requires `M % 128 == 0`, but inference uses batch sizes 1-24. For decode (M=1-24), the kernel can't be used directly. It would work for **prefill** (prompt length typically > 128).

### Next Step

Fix the kernel to handle small M by either:
- Using smaller MMA tile configurations (m16n8k32 handles M≥16 with padding)
- Adding boundary masking for partial tiles
- Dispatching fused kernel only for prefill, Marlin for decode

Shall I work on the small-M fix next?
