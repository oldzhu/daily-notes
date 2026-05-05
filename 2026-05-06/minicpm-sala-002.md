Searched for regex `w4a8|W4A8|machete|Machete|qqq|QQQ` (`**/sgl-kernel/**`), 20 results

Searched for regex `sm_|SM_|__CUDA_ARCH__|compute capability|sm120|sm90|sm80` (`**/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu`), 9 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/tests/test_qserve_w4a8_per_group_gemm.py#1-1), lines 1 to 60

Read [](file:///home/oldzhu/sglang/sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py#1-1), lines 1 to 50

Searched for regex `fp8|FP8|e4m3|float_e4m3|sm_|SM_|90a|hopper|blackwell|sm120` (`**/sgl-kernel/csrc/moe/cutlass_moe/w4a8/**`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py#50-50), lines 50 to 120

Let me verify vllm Machete W4A8 status:  



Fetched 2 resources

Now I have clear answers. Let me update the proposal doc with the findings, then answer.   

Searched for regex `getSMVersion|sm_version|>=\s*\d+|<=\s*\d+|TORCH_CHECK` (`**/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu`), 15 results

Now I have the full picture. Let me answer each question directly.

---

## Answers

### 1) "Approve if better on both memory and compute"

Understood — both memory and compute wins are required, otherwise we keep Marlin W4A16. Phase 0 microbenchmark must confirm this **before** any kernel integration work.

### 2) Existing W4A8 kernels — pros / cons / what's already in sgl-kernel

**Important finding**: I checked sgl-kernel and vllm. Two relevant kernels exist:

| Kernel | Location | Weight | Activation | MMA | Arch | Status |
|---|---|---|---|---|---|---|
| **`qserve_w4a8_per_group_gemm`** + `per_chn` | sgl-kernel `csrc/gemm/` (already vendored, MIT QServe) | INT4 | **INT8** | I8 IMMA | SM80+ → **should run on SM120** | dense GEMM, ready to integrate |
| **`cutlass_w4a8_moe_mm`** | sgl-kernel `csrc/moe/cutlass_moe/w4a8/` | INT4 | **FP8 e4m3** | FP8 QMMA (via Hopper TMA) | **SM90 only** | MoE-only (grouped GEMM), NOT dense |
| **vllm Machete** | vllm `csrc/quantization/machete/` | INT4 | **BF16/FP16** (NOT FP8) | BF16 MMA | SM90 (Hopper-optimized) | Mixed-precision, **W4A16 not W4A8** |

Key facts:
- **vllm Machete is W4A16, not W4A8.** Its `compute_type = a.dtype`, where a is BF16/FP16. It's a Hopper successor to Marlin for **the same problem space we already cover** with Marlin SM120. It does NOT do FP8 activation by default. (Some downstream forks added FP8, but not in mainline.)
- **Dense W4A8-FP8 on SM120 has no existing kernel**, anywhere I can find. The only existing W4-FP8 kernel in sgl-kernel/vllm is the SM90 MoE one.
- **Dense W4A8-INT8 on SM120: kernel already exists in sgl-kernel** (QServe). SM_ARCH ≥ 80 path uses inline PTX I8 IMMA. It just needs Python wiring in `gptq.py` + weight repack + activation INT8 quantizer.

This **flips the effort calculus**:

| Path | Effort | Compute risk on SM120 | Memory win |
|---|---|---|---|
| **Option A: W4-FP8 dense** | **HIGH** — write/port a new kernel from scratch (or adapt the SM90 MoE one to dense + SM120). Possibly weeks. | LOW — FP8 QMMA listed at 296 TF | YES (INT4 weight retained + FP8 activation) |
| **Option B: W4-INT8 dense (QServe)** | **LOW** — kernel already in sgl-kernel; only Python integration + weight repack | UNKNOWN — INT8 IMMA throughput on SM120 not in official spec; needs Phase 0 microbenchmark | YES (INT4 weight retained + INT8 activation) |
| **Option C: custom CUTLASS** | HIGHEST | varies | varies |

Per your point #4 ("no champion has tried real W4A8-FP8 in past week → unproven path"): combined with the no-dense-kernel reality, **Option A becomes a research project, not an iteration.** The pragmatic choice is now:

- **Recommended**: Run Phase 0 INT8 IMMA microbench. If passes → do **Option B (QServe W4-INT8)** which is a low-risk, ready-kernel integration. If Phase 0 fails → no real W4A8 path is cheap on SM120; defer the whole topic.

### 3) Phase 0 microbenchmark — commands & script

Yes please launch fcloud when you're ready. Test plan:

```bash
# On fcloud, after `python3 scripts/fcloud/fcloud_workflow.py setup`:
ssh-style or via fcloud_exec, in /root:

# Quick microbench using PyTorch (no cutlass_profiler build needed)
cat > /root/bench_int8_vs_fp8_sm120.py <<'PY'
import torch, time
torch.manual_seed(0)
DEV = "cuda"

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# Representative MiniCPM shapes (qkv_proj at 4096x4096, gate_up at 4096x14336)
shapes = [(4096, 4096, 4096), (4096, 14336, 4096), (4096, 4096, 14336)]
for M, N, K in shapes:
    a_bf16 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    a_fp8  = a_bf16.to(torch.float8_e4m3fn)
    b_fp8  = b_bf16.to(torch.float8_e4m3fn)
    a_i8   = (a_bf16 * 100).clamp(-128, 127).to(torch.int8)
    b_i8   = (b_bf16 * 100).clamp(-128, 127).to(torch.int8)
    scale_a = torch.tensor(1.0, device=DEV); scale_b = torch.tensor(1.0, device=DEV)

    t_bf16 = bench(lambda: torch.matmul(a_bf16, b_bf16))
    t_fp8  = bench(lambda: torch._scaled_mm(a_fp8, b_fp8.t().contiguous().t(),
                       scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16))
    try:
        t_i8 = bench(lambda: torch._int_mm(a_i8, b_i8))
        i8_str = f"INT8={t_i8*1e3:.3f}ms ({2*M*N*K/t_i8/1e12:.1f} TFLOPS)"
    except Exception as e:
        i8_str = f"INT8=N/A ({e})"

    flops = 2 * M * N * K
    print(f"[{M}x{N}x{K}]  BF16={t_bf16*1e3:.3f}ms ({flops/t_bf16/1e12:.1f} TF)  "
          f"FP8={t_fp8*1e3:.3f}ms ({flops/t_fp8/1e12:.1f} TF)  {i8_str}")
PY

python3 /root/bench_int8_vs_fp8_sm120.py 2>&1 | tee /root/phase0_int8_vs_fp8_sm120.log
```

Decision rule (results recorded into `docs/soar_2026_changes/PHASE0_INT8_IMMA_SM120.md` after run):
- **INT8 ≥ 250 TF (≈ FP8)** → green-light Option B (QServe W4-INT8 integration).
- **INT8 ≈ 148 TF (BF16 rate)** → drop INT8; abandon Option B.
- **FP8 < 250 TF** → check spec assumption; investigate why.

### 4) On Option A being unproven

You're right. Combined with the finding that **no dense W4A8-FP8 kernel exists for SM120**, Option A is a research project. Better strategy:
1. Run Phase 0 (cheap, ~5 min on fcloud).
2. If Phase 0 says INT8 IMMA works → do Option B (QServe is ready, just integrate).
3. If both 8-bit paths look unfavorable on SM120 → drop W4A8 entirely and pivot to other catalog items (speculative decoding tuning, attention kernel improvements, etc.).

---

**Ready when you are.** Once you start fcloud, I will:
1. Run the Phase 0 microbench above
2. Save log and create `docs/soar_2026_changes/PHASE0_INT8_IMMA_SM120.md` with results
3. Shut down fcloud
4. Recommend next step based on data

Want me to start when fcloud is up?  


