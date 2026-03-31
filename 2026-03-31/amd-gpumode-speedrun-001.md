I’m grounding this in the kernels we already observed from remote runs, so I can answer with the actual backend behavior of your current three submissions rather than generic AITER theory.

Searched for regex `module_mla_asm|module_gemm_a4w4_asm|module_moe_ck2stages|FlyDSL|CK|asmGEMM|CKGEMM` (`**/*.{md,py}`), 87 results

Read [](file:///home/oldzhu/gpumode/amd202602_vs/AGENTS.md)

Read [](file:///home/oldzhu/gpumode/amd202602_vs/moe-mxfp4/submission_clean.py)

There is no single global default backend in AITER. It dispatches per operator, shape, dtype, and what optional backends are available.

For your current three submissions, the remote logs show:

- submission_clean.py is currently using an ASM-backed MLA path.
  The runs loaded `module_mla_asm` and a kernel like `mla_a8w8_... .co`.

- submission_clean.py is currently using an ASM-backed GEMM path.
  The runs loaded `module_gemm_a4w4_asm` and kernels like `f4gemm_bf16_per1x32Fp4_BpreShuffle_... .co`.

- submission_clean.py is currently using a CK-backed FusedMoE path.
  The runs loaded `module_moe_ck2stages_...`, and AITER’s README says FusedMoE can use FlyDSL optionally, otherwise it falls back to CK.

So the answer to “is the current default CK?” is: not generally. In this repo right now it is:
- mixed-mla: ASM
- mxfp4-mm: ASM
- moe-mxfp4: CK

On “how do I use Triton or ASM instead”:

1. AITER auto-selects a backend internally when you call a high-level op like `mla_decode_fwd`, `gemm_a4w4`, or `fused_moe`.
2. For some ops, installing optional dependencies changes what AITER can choose.
3. If you want full control, you usually stop using the high-level AITER op and call your own kernel directly.

The most realistic split is:

- FlyDSL:
  Mostly relevant to FusedMoE.
  AITER explicitly says FusedMoE supports FlyDSL-based kernels and falls back to CK when FlyDSL is unavailable.

- Triton:
  Best when you want to write a custom kernel in Python and let Triton lower it to AMD GPU code.
  Good for trying specialized kernels or fusion.

- ASM:
  Best ceiling, worst development speed.
  This is the route for exact low-level control, but it is much harder to build and tune.

Important nuance: Triton and FlyDSL do generate GPU machine code, but they do not give you direct instruction-by-instruction control the way handwritten ASM does. They are compiler-driven. You can get better performance, but only if your generated kernel is better tuned for the exact workload than AITER’s current kernel.

**How to try FlyDSL with AITER**
For MoE, the least invasive path is to keep submission_clean.py mostly unchanged and change the runtime environment so AITER can pick a FlyDSL backend.

Example:

```bash
pip install --pre flydsl
pip install -r requirements.txt
```

Then run the same submission again and inspect stderr/profile output. If AITER is still loading `module_moe_ck2stages...`, you are still on CK. If the backend changes, it should show up in the loaded module names or profile traces.

In other words, this code can stay the same:

```python
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

output = fused_moe(
    hidden_states,
    gate_up_weight_shuffled,
    down_weight_shuffled,
    topk_weights,
    topk_ids,
    expert_mask=None,
    activation=ActivationType.Silu,
    quant_type=QuantType.per_1x32,
    doweight_stage1=False,
    w1_scale=gate_up_weight_scale_shuffled,
    w2_scale=down_weight_scale_shuffled,
    a1_scale=None,
    a2_scale=None,
    hidden_pad=hidden_pad,
    intermediate_pad=intermediate_pad,
)
```

If FlyDSL support is active for that operator, AITER should pick it under the hood.

**How to use Triton directly**
This means bypassing the AITER operator and calling your own Triton kernel from the submission file.

A minimal shape of that looks like this:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def launch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

For your problems, the direct-Triton route would mean replacing:
- `gemm_a4w4` in submission_clean.py, or
- part of the MLA or MoE pipeline

with a custom Triton kernel.

Practically, mxfp4-mm is the most realistic Triton target because it is structurally simpler than MLA and MoE. But to be competitive, that Triton kernel would need to handle:
- packed MXFP4 data
- E8M0 scales
- preshuffled layouts
- tile tuning for MI355X

That is real kernel work, not just swapping APIs.

**How to use ASM directly**
This is usually not “write a few lines in Python.” It is more like:

1. write or obtain an AMD GPU kernel in assembly or HIP/LLVM IR
2. compile it into HSACO
3. launch it from a Python extension, PyTorch custom op, or an AITER-like wrapper

Conceptually:

```python
# Pseudocode only
from my_custom_extension import launch_my_hsaco_kernel

def custom_kernel(data):
    A, B, scales = ...
    out = torch.empty(...)
    launch_my_hsaco_kernel(A, B, scales, out)
    return out
```

This is the highest ceiling, but also the slowest path unless you already have AMD internal kernel help or a strong existing template.

**Practical recommendation**
For this repo, I would not try handwritten ASM first.

The order with the best expected ROI is:

1. moe-mxfp4 with FlyDSL-backed FusedMoE, because AITER already supports that path conceptually.
2. mxfp4-mm with a direct Triton experiment, if we decide to replace `gemm_a4w4` or fuse A-side quantization.
3. only consider raw ASM if we already know exactly which kernel shape/path we want and have support to build it.

If you want, I can do the next step in a concrete way:
1. inspect how to tell whether AITER is actually using FlyDSL for MoE and whether we can force it
2. draft a Triton prototype plan for mxfp4-mm
3. look through AITER’s public source surface for backend-selection knobs rather than relying on stderr inference
