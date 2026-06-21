Let me read the full kernel before deciding: 

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu)

Autopilot recovered from a request errorCompacted conversationAutopilot recovered from a request errorCompacted conversation

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#78-78), lines 78 to 145

The root cause is confirmed. Three related bugs in Phase 3 and the epilogue. Here is my proposal, per SOAR workflow — please review before I touch the code.

---

## Proposal: Fix MMA Fragment Layout (Bug 4)

### Root Cause

For `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, PTX ISA specifies exact per-thread element ownership. All three current load/store patterns violate that spec.

**A-operand (16×32 FP8, row-major, 4 regs)** — thread `t` must own:

| reg | row | K-cols |
|-----|-----|--------|
| `a[0]` | `t/4` | `(t%4)*4 + 0..3` |
| `a[1]` | `t/4` | `(t%4)*4 + 16..19` |
| `a[2]` | `t/4 + 8` | `(t%4)*4 + 0..3` |
| `a[3]` | `t/4 + 8` | `(t%4)*4 + 16..19` |

Current code uses `r*8` row-stride (→ rows 0,8,16,24) and `(t%4)*8` column-stride — both wrong.

**B-operand (K×N = 32×8, col-major, 2 regs)** — thread `t` must own:

| reg | N-col | K-rows |
|-----|-------|--------|
| `b[0]` | `t/4` | `(t%4)*4 + 0..3` |
| `b[1]` | `t/4` | `(t%4)*4 + 16..19` |

Current code uses N-col `(t%4)*2` and K-stride `t/4 + r*16` — both wrong.

**C/D accumulator (16×8 FP32, 4 regs)** — thread `t` writes:

| reg | global row | global col |
|-----|-----------|-----------|
| `cp[0]` | `m0+wm+t/4` | `n0+wn+2*(t%4)` |
| `cp[1]` | `m0+wm+t/4` | `n0+wn+2*(t%4)+1` |
| `cp[2]` | `m0+wm+t/4+8` | `n0+wn+2*(t%4)` |
| `cp[3]` | `m0+wm+t/4+8` | `n0+wn+2*(t%4)+1` |

Current epilogue uses `(t*4+i)/8` / `(t*4+i)%8` — wrong (linear traversal instead of row-paired layout).

### Files to change

w4a8_fp8_qmma.cu — Phase 3 inner loops (~30 lines) + epilogue (~10 lines)

### Validation

Expected: correctness test reports `mean ≈ 768.0` (currently `9600.0`, 12.5× too large).

### Risk

Low — pure data-layout fix, no algorithmic change, no impact on accuracy/speed until the kernel actually produces correct results.

---

Shall I apply the fix?
