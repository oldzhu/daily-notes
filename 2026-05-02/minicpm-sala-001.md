Compacted conversationI'll draft the M1 proposal docs. Let me first inspect the current Marlin dispatch code and CHANGE_0125 to ground the proposal in reality.   

Searched for files matching `**/gptq_marlin.cu`, 1 match

Searched for files matching `**/CHANGE_0125*`, 4 matches

Searched for files matching `**/OPTIMIZATION_CATALOG*`, 1 match

Searched for regex `determine_exec_config|exec_config|atomic_add|use_atomic_add|thread_k_blocks|thread_n_blocks` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#180-180), lines 180 to 320

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#130-130), lines 130 to 180

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#640-640), lines 640 to 850

Searched for regex `use_atomic_add|16384|ceil.*prob_m.*prob_n` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), 9 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0125_sm120_marlin_tiles_001.en.md#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0125_sm120_marlin_tiles_001.en.md#80-80), lines 80 to 182

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1100-1100), lines 1100 to 1160

## Important finding before drafting — must revise M1

While inspecting gptq_marlin.cu and re-reading CHANGE_0125_sm120_marlin_tiles_001.en.md, I discovered we **already conducted and concluded** the per-shape tile investigation I was about to re-propose. Key findings from that doc (Test 27):

1. The SM120 scorer selects narrow-tile `{4,8,128}` for **every** MiniCPM-SALA linear shape, not by accident — `fill_ratio × 1000` dominates and narrow tiles maximize SM occupancy on our N dimensions (1024–28672).
2. Wide tiles (`thread_n=256`) would need N ≥ 24,576 just to fill 96 SMs, and even at N=28672 narrow tiles still win on `wave_ratio`.
3. **"Tuning the scoring function is a dead end"** and **"more tile instantiations don't help"** — explicitly documented conclusions.
4. The file recommends SM120-native MMA / TMA / QMMA / CUTLASS 3.x as the real path.

So the M1 I sketched ("per-shape tile dispatch per FlashSALA W6 blog") would largely **repeat a dead-end path** we already burned a test cycle on. FlashSALA's actual gain must be coming from deeper-than-tile changes (likely CUTLASS 3.x SM120 GEMM or QMMA mxfp8).

What **is still unexplored** in the Marlin file and could be a genuine quick win:

### The only remaining Marlin-level lever — atomic_add guard (line 1142)

```cpp
bool part_use_atomic_add = use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;
```

For MiniCPM-SALA shapes:
- Smallest: `z_proj` N=1024 → `ceil(1/64)·1024 = 1024 ≤ 2048` ✓ atomic_add used
- `lightning o_proj` N=4096 → `1·4096 = 4096 > 2048` ✗ atomic_add **disabled** for M=1 decode
- `std qkv_proj` N=6144 → `6144 > 2048` ✗ **disabled**
- `gate_up_proj` N=28672 → `28672 > 2048` ✗ **disabled**

At M=1 (decode), 98% of our GEMMs fall into the "barrier path" instead of atomic_add. FlashSALA's W6 post specifically mentions relaxing this for small-M. Raising the threshold to e.g. 32768 (or dropping the guard for `prob_m_split ≤ 32`) is a genuine, untested, additive experiment.

---

## Revised proposal

### **M1-atomic (LOW risk, 0.5–1 day, expected +1–5% decode)** — atomic_add guard relaxation
- File: gptq_marlin.cu line 1142
- Change: `part_use_atomic_add = use_atomic_add && (prob_m_split <= 32 || div_ceil(prob_m_split, 64) * prob_n <= 2048)`
- Rebuild incremental (~5 min with ccache), smoke-test acc, run Test 12 speed.
- Real gain is small; value is **finishing the Marlin investigation**.

### **M1-real — SM120-native GEMM via CUTLASS 3.x or QMMA mxfp8** (HIGH effort, 2–3 weeks, expected +30–80%)
Per CHANGE_0125_001's own recommendation, this is where FlashSALA's advantage almost certainly lives:
- **Option A**: integrate CUTLASS 3.x `sm120` GEMM kernel for W4A16 (replace Marlin call path for inference)
- **Option B**: introduce a mxfp8 (QMMA) GEMM path — 296 TFLOPS vs Marlin's ~148 TFLOPS BF16 ceiling
- High uncertainty on integration effort; needs a 2-day spike to assess feasibility before committing

### **M2 unchanged — Mixed-precision KV cache (FP8 edges + FP4 middle)**
Still the biggest clean win for Smax; Week 5 semifinal winner's playbook.

### **M3 unchanged — Official-aligned local speed benchmark**

---

## Recommended action

Given CHANGE_0125_001's findings, I'd rather **not** write a proposal doc that rehashes the tile-dispatch idea. Instead I propose:

1. **Today / next turn**: draft a **short M1-atomic proposal** (half-day test; closes the Marlin chapter)
2. **In parallel**: draft a **SM120 kernel spike plan** — 2-day investigation into CUTLASS 3.x sm120 GEMM & mxfp8 feasibility, to decide if M1-real is tractable
3. **Then**: M2 mixed-precision KV (the surest large gain)
4. **Then**: M3 speed-set rebuild

Do you want me to:
- (a) draft `PROPOSAL_iteration_M1_atomic_add_guard.{en,zh}.md` (short, actionable), **and**
- (b) draft `PROPOSAL_sm120_gemm_spike.{en,zh}.md` (investigation plan, no code change yet)?

Or would you prefer to skip Marlin entirely and go straight to M2 (mixed-precision KV) proposal? 


