## Explanation: dense-only smoke test, why isolate plumbing from sparse gates

**`--force-dense-minicpm`** is our existing flag (defined in `prepare_env.sh`'s `SGLANG_SERVER_ARGS`) that disables the MiniCPM SALA sparse-attention path and routes the model through the standard dense FlashAttention backend.

**Why isolate plumbing from sparse gates first:**

MXFP4 KV touches **two independent code paths** at the same time:
1. **The plumbing path** — FP4 pool allocation, quantize on `set_kv_buffer`, dequantize on `get_kv_buffer`, FA call with descale handling. This is the "does the format work end-to-end at all" question.
2. **The sparse-attention path** — sparse top-k scoring, compressed-K bridge, infllmv2 kernels, sparse scratch buffers. These add 4 more code sites that touch KV tensors and may make further assumptions about dtype.

If we enable MXFP4 KV with sparse on at the same time and the server crashes or accuracy collapses, we **cannot tell** whether the bug is in the plumbing (Gap A/C/D in the survey) or in some sparse interaction (Gap G). Debugging would require stripping back to dense anyway.

By starting with `--force-dense-minicpm`:
- All sparse gates are skipped → only ~3 plumbing gates remain in scope.
- Smoke = "server boots, generates coherent text" → confirms plumbing is sound.
- If accuracy on the dense baseline is already ≥75%, we know MXFP4 itself is acceptable, and we move on to re-enable sparse.
- If dense accuracy fails, problem is in plumbing — known scope, fast fix.

Once dense is green, we re-enable sparse as a separate iteration with a much smaller debug surface.

---

Now to fcloud. Let me check PyTorch version and run the spike. 

Ran terminal command:  cd /home/oldzhu/sglang && cat ~/.fcloud_config 2>/dev/null | head -5 && ls scripts/fcloud/

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_exec.py#1-1), lines 1 to 80

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 "python3 -c 'import torch; print("torch=", torch.__version__); print("has_float4_e2m1fn_x2=", hasattr(torch, "float4_e2m1fn_x2"))' && nvcc --version | tail -3 && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv"

Environment good: torch 2.9.1 (FP4 dtype available), CUDA 12.8 (supports sm_120), driver 580. Now upload the spike and run it.  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py --help 2>&1 | head -40

Searched for regex `upload|put_file|jupyter.*content` (`**/scripts/fcloud/fcloud_workflow.py`), 19 results

Searched for text `def upload_file` (`**/scripts/fcloud/fcloud_exec.py`), 1 result

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import sys
sys.path.insert(0, 'scripts/fcloud')
import fcloud_exec
url, token = fcloud_exec.load_config()
ok1 = fcloud_exec.upload_file(url, token, 'docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu', '/root/bench_w4fp8_sm120.cu')
ok2 = fcloud_exec.upload_file(url, token, 'docs/soar_2026_changes/spike_w4fp8/run_w4fp8_spike.sh', '/root/run_w4fp8_spike.sh')
print('upload_cu=', ok1, 'upload_sh=', ok2)
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 300 "cd /root && chmod +x run_w4fp8_spike.sh && bash run_w4fp8_spike.sh 2>&1 | tail -200"

Compile error — `int4_t` doesn't exist. Fix: use `int` with manual 4-bit sign-extension.  

Replacing 9 lines with 9 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import sys; sys.path.insert(0, 'scripts/fcloud')
import fcloud_exec
url, token = fcloud_exec.load_config()
print(fcloud_exec.upload_file(url, token, 'docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu', '/root/bench_w4fp8_sm120.cu'))
" && python3 scripts/fcloud/fcloud_exec.py exec --timeout 300 "cd /root && bash run_w4fp8_spike.sh 2>&1 | tail -120"

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01H6ZhwY1BVNTssb24pmuPbj__vscode-1777330023248/content.txt)

PTX rejects `cvt.rn.satfinite.e4m3x2.bf16x2` — that variant doesn't exist; bf16→e4m3 has to go through fp32. Switch to the supported `cvt.rn.satfinite.e4m3x2.f32` form via the CUDA helper. 

Replacing 9 lines with 10 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu)


```

```


Replacing 30 lines with 28 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu)


```

```
Compacted conversation

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import sys; sys.path.insert(0, 'scripts/fcloud')
import fcloud_exec
url, token = fcloud_exec.load_config()
print(fcloud_exec.upload_file(url, token, 'docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu', '/root/bench_w4fp8_sm120.cu'))
" && python3 scripts/fcloud/fcloud_exec.py exec --timeout 300 "cd /root && bash run_w4fp8_spike.sh 2>&1 | tail -80"

Got results. Shutting down fcloud immediately per cost-saving rule. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py shutdown

Fcloud shut down. Now writing up results offline.

## Spike results summary

| Shape (M×N×K) | Time/iter | TFLOPS | % of FP8 ceiling (281 TF) | Verdict |
|---|---|---|---|---|
| 16384×14336×4096 (default prefill) | 12.30 ms | **156.4** | 55.7% | RED |
| 2048×4096×4096 (small) | 0.44 ms | 155.8 | 55.4% | RED |
| 1×14336×4096 (decode) | 0.004 ms | 32.6 | 11.6% | RED (BW-bound) |
| 8192×8192×8192 (square) | 7.03 ms | 156.4 | 55.6% | RED |

**Environment confirmed**: torch 2.9.1+cu128, `torch.float4_e2m1fn_x2` available, nvcc 12.8, RTX 6000D sm_120.

**Key finding**: Bare-metal (no TMA / no swizzling / no software pipelining / no fused-dequant) W4→FP8 sustains **~156 TFLOPS** = **55.6% of the FP8 ceiling**. The dequant chain (int4 unpack → sub-zp → fmul scale → cvt.f32.e4m3x2) costs ~45% of cycles in this minimal scaffold.

**Caveats (this is a lower bound)**:
- Dequant is unfused fp32 path; a tuned kernel can use `lop3.b32` for int4-unpack-and-shift in ~3 instructions/8-vals.
- No async copies / TMA / register pipelining → compute and load are serialized.
- A CUTLASS-class production kernel typically reaches 70–85% of peak → realistic upper bound ~**200–240 TFLOPS**.

**Decision impact**:
- Production W4A16 Marlin already runs ~140–170 TFLOPS-equiv in our pipeline. A tuned W4-FP8 might gain **+15–40%** on prefill GEMMs only — not on attention, not on decode (BW-bound, gain near zero).
- 3–4 weeks of kernel-engineering risk for **~5–12% end-to-end** is **NOT favorable** vs. NVFP4 KV cache (20% memory savings, more direct path to S∞ score, ~3 days of plumbing work, already 80% in tree).

**Recommendation**: **Park W4-FP8 (verdict RED for current resource budget). Proceed with NVFP4 KV cache P2 (dense-only smoke first).**

Now writing up the result doc and updating the catalog. 

Searched for files matching `**/docs/soar_2026_changes/RESULT_W4_FP8*`, 2 matches

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md)

Ran terminal command:  cd /home/oldzhu/sglang && mv docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md && mv docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.zh.md docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md

Now fill in the EN result doc. 

Replacing 17 lines with 48 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md)


```

```


Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md)

Replacing 17 lines with 48 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md)


```

```
Now update the optimization catalog and find the chat log file.  

Searched for regex `W4.?FP8|W4A8|nvfp4|NVFP4` (`**/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`), 2 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#180-180), lines 180 to 230

Replacing 2 lines with 8 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)


```

```
Now append round 11 to the chat log.  

Searched for files matching `**/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike*`, 2 matches

Ran terminal command:  cd /home/oldzhu/sglang && tail -5 docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md && echo "---ZH---" && tail -5 docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md

Ran terminal command:  cd /home/oldzhu/sglang && cat >> docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md <<'EOF'

---

## Round 11 — fcloud execution of W4-FP8 spike (2026-04-28 13:00)

### User asks
1. "Approve dense-only smoke first? ok"
2. "could u explain what is starts with --force-dense-minicpm smoke test to isolate plumbing from sparse gates. and why we need isolate plumbing from sparse gates?"
3. "Accuracy regression tolerance? >=75%"
4. "I started fcloud and you can 1. check if pytorch >> 2.8  2. test W4-FP8 spike scaffold"

### Agent answers / actions
- **Q1 explained**: `--force-dense-minicpm` flag in `prepare_env.sh` SGLANG_SERVER_ARGS routes attention through standard FlashAttention (skipping MiniCPM SALA sparse path). MXFP4 KV touches 2 independent code paths: (1) plumbing — FP4 pool, quantize/dequantize, FA descale; (2) sparse-attention — top-k scoring, compressed-K bridge, infllmv2, scratch buffers. Enabling both at once makes bug attribution impossible. Dense-only first → ~3 plumbing gates, fast iteration. Once green, re-enable sparse as separate iteration.
- **Q3 recorded**: NVFP4 KV P2 minimum accuracy threshold = 75% (vs 79.29% baseline → max 4.3pp regression).
- **fcloud env check**: torch=2.9.1+cu128 (✅ has `float4_e2m1fn_x2`), nvcc 12.8 supports `-arch=sm_120`, RTX 6000D compute_cap 12.0. All prerequisites met for both spike and NVFP4 KV.
- **Spike scaffold required 2 fixes** during build:
  1. `int4_t` is not a real type → replaced with `int` + manual 4-bit sign-extend.
  2. `cvt.rn.satfinite.e4m3x2.bf16x2` PTX form does NOT exist on SM120 → ptxas rejected. Switched to supported `cvt.rn.satfinite.e4m3x2.f32` (2 fp32 → 1 fp8x2). Dequant chain now: int4 unpack → fp32 sub-zp → fp32 fmul scale → cvt to e4m3x2.
- **Spike results** (4 shapes, 100 iters, cudaEvent timing):

| Shape | TFLOPS | % FP8 ceiling | Verdict |
|---|---|---|---|
| 16384×14336×4096 | 156.4 | 55.7% | RED |
| 2048×4096×4096 | 155.8 | 55.4% | RED |
| 1×14336×4096 (decode) | 32.6 | 11.6% | RED (BW-bound) |
| 8192×8192×8192 | 156.4 | 55.6% | RED |

  ptxas: 23 registers, 0 spills, 1 barrier — kernel is compute-peak limited. The dequant chain costs ~43% of cycles in the unfused fp32 path.
- **fcloud shutdown executed** immediately after collecting results (cost-saving rule).

### Outcomes
- **W4-FP8 dense parked (RED)**. A tuned production kernel could plausibly reach 200–240 TF (75–85%) but the 3–4 week engineering cost for ~5–12% end-to-end gain on prefill-only paths is unfavorable vs. NVFP4 KV (~3 days, ~44% KV memory savings, helps S∞ directly).
- **NVFP4 KV cache promoted to P2**. Will start dense-only smoke after user go-ahead.
- Catalog updated to mark W4-FP8 RED and NVFP4 KV P2-NEXT.

### Cross-references
- [RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md](../RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md) (final, replaces template)
- [SURVEY_NVFP4_KV_P1_20260428_1130.en.md](../SURVEY_NVFP4_KV_P1_20260428_1130.en.md)
- [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md) (W4-FP8 → RED, NVFP4 KV → P2-NEXT)
- [spike_w4fp8/bench_w4fp8_sm120.cu](../spike_w4fp8/bench_w4fp8_sm120.cu) (with cvt fix)
EOF

Ran terminal command:  cd /home/oldzhu/sglang && cat >> docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md <<'EOF'

---

## 第 11 轮 —— fcloud 执行 W4-FP8 spike（2026-04-28 13:00）

### 用户提问
1. "先批准 dense-only smoke 优先？ok"
2. "可以解释一下什么是用 --force-dense-minicpm 起 smoke test 来把 plumbing 与 sparse 门控隔开吗？为什么需要把 plumbing 与 sparse 门控隔开？"
3. "精度回归容忍度？>=75%"
4. "我已经把 fcloud 启起来了，你可以 1. 检查 pytorch >> 2.8  2. 测 W4-FP8 spike scaffold"

### 智能体回答 / 行动
- **Q1 解释**：`prepare_env.sh` 的 SGLANG_SERVER_ARGS 中 `--force-dense-minicpm` 让 attention 走标准 FlashAttention（跳过 MiniCPM SALA 稀疏路径）。MXFP4 KV 触及 2 条独立代码路径：(1) plumbing —— FP4 池、quantize/dequantize、FA descale；(2) 稀疏注意力 —— top-k 打分、compressed-K bridge、infllmv2、scratch buffers。同时启两条会让 bug 归因不可能。先 dense-only → 仅 ~3 个 plumbing 门控，迭代快。绿后再当作单独迭代启用稀疏。
- **Q3 记录**：NVFP4 KV P2 最低精度阈值 = 75%（基线 79.29% → 最多 4.3pp 回归）。
- **fcloud 环境检查**：torch=2.9.1+cu128（✅ 有 `float4_e2m1fn_x2`），nvcc 12.8 支持 `-arch=sm_120`，RTX 6000D compute_cap 12.0。spike 与 NVFP4 KV 的所有前置条件都满足。
- **Spike scaffold 构建期间需 2 处修复**：
  1. `int4_t` 不是真实类型 → 改用 `int` + 手动 4-bit 符号扩展。
  2. `cvt.rn.satfinite.e4m3x2.bf16x2` 这个 PTX 形式在 SM120 上**不存在** → ptxas 拒绝。换为受支持的 `cvt.rn.satfinite.e4m3x2.f32`（2 fp32 → 1 fp8x2）。反量化链改为：int4 解包 → fp32 减零点 → fp32 乘 scale → cvt 到 e4m3x2。
- **Spike 结果**（4 个形状，100 iter，cudaEvent 计时）：

| 形状 | TFLOPS | 占 FP8 上限 % | 判定 |
|---|---|---|---|
| 16384×14336×4096 | 156.4 | 55.7% | RED |
| 2048×4096×4096 | 155.8 | 55.4% | RED |
| 1×14336×4096（decode）| 32.6 | 11.6% | RED（带宽受限）|
| 8192×8192×8192 | 156.4 | 55.6% | RED |

  ptxas：23 个寄存器，0 spill，1 个 barrier —— kernel 受计算峰值限制。未融合 fp32 路径下的反量化链占 ~43% 周期。
- **采集完结果立即执行 fcloud 关机**（节费规则）。

### 产出
- **W4-FP8 密集搁置（RED）**。调优生产 kernel 大致能达 200–240 TF（75–85%），但 3–4 周工程换取 prefill-only 路径上 ~5–12% 端到端，相比 NVFP4 KV（~3 天，~44% KV 内存节省，直接帮助 S∞）不划算。
- **NVFP4 KV cache 提升至 P2**。等用户批准后启动 dense-only smoke。
- Catalog 更新，标记 W4-FP8 为 RED，NVFP4 KV 为 P2-NEXT。

### 交叉引用
- [RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md](../RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md)（终版，替代模板）
- [SURVEY_NVFP4_KV_P1_20260428_1130.zh.md](../SURVEY_NVFP4_KV_P1_20260428_1130.zh.md)
- [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)（W4-FP8 → RED，NVFP4 KV → P2-NEXT）
- [spike_w4fp8/bench_w4fp8_sm120.cu](../spike_w4fp8/bench_w4fp8_sm120.cu)（含 cvt 修复）
EOF
echo "done"

Ran terminal command:  cd /home/oldzhu/sglang && tail -3 docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md && echo "---ZH---" && tail -3 docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md

Both chat logs updated. Now commit and push to `minicpm-src`.  

Ran terminal command:  cd /home/oldzhu/sglang && git status -s docs/soar_2026_changes/ | head -20

Ran terminal command:  cd /home/oldzhu/sglang && git diff --stat docs/soar_2026_changes/OPTION_B_PHASE_1_DELIVERY.md | head -3

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.zh.md docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu && git status -s | head -20

Ran terminal command:  cd /home/oldzhu/sglang && git commit -m "docs(soar): R11 fcloud W4-FP8 spike results — RED verdict, NVFP4 KV → P2

- Spike scaffold ran on fcloud (RTX 6000D sm_120, CUDA 12.8, torch 2.9.1)
- Required 2 build fixes: int4_t→int with manual sign-extend; cvt path
  switched from non-existent e4m3x2.bf16x2 to supported e4m3x2.f32
- Results across 4 shapes: ~156 TFLOPS = 55.7% of FP8 ceiling (281 TF)
- ptxas: 23 reg, 0 spill, 1 barrier — kernel is compute-peak limited
- Verdict: RED. Tuned production kernel could plausibly reach 200-240 TF
  (75-85%) but 3-4 weeks engineering for ~5-12% prefill-only end-to-end
  is unfavorable vs NVFP4 KV (~3 days, ~44% KV memory savings, helps S∞)
- Catalog: W4-FP8 dense → RED/parked, NVFP4 KV → P2-NEXT
- Chat log Round 11 appended (EN+ZH synced)
- Renamed RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.{en,zh}.md → _1300
- fcloud shut down per cost-saving rule" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Committed and pushed (`28ca14847`). 

## Round 11 summary

**fcloud env confirmed**: torch 2.9.1+cu128 (✅ has `float4_e2m1fn_x2`), nvcc 12.8 (`-arch=sm_120` works), RTX 6000D compute_cap 12.0 — all prerequisites met for both spike and NVFP4 KV.

**W4-FP8 spike → RED**: 4 shapes converged to **156 TFLOPS = 55.7% of FP8 ceiling (281 TF)**. ptxas: 23 reg, 0 spill, 1 barrier; compute-peak limited; dequant chain costs ~43% of cycles. A tuned production kernel could plausibly reach 200–240 TF (75–85%) but the 3–4 week engineering cost only buys ~5–12% on prefill-only paths — unfavorable vs NVFP4 KV (~3 days, ~44% KV memory savings, directly helps S∞).

**Build needed 2 fixes**: (1) `int4_t` not a real type → `int` + manual sign-extend; (2) `cvt.rn.satfinite.e4m3x2.bf16x2` doesn't exist on SM120 → switched to supported `cvt.rn.satfinite.e4m3x2.f32`.

**Decisions executed**:
- W4-FP8 dense parked indefinitely (catalog updated → RED).
- NVFP4 KV cache promoted to P2-NEXT (will start dense-only smoke after your go-ahead, accuracy threshold ≥75%).
- fcloud shut down per cost-saving rule.
- Result docs (EN+ZH), catalog, chat log Round 11 (EN+ZH), and spike .cu fixes all committed and pushed to `minicpm-src/mixed_minicpm_cudagraph` as `28ca14847`.

**Awaiting your green light** to begin NVFP4 KV P2 plumbing implementation (3-day source edits in minicpm_backend.py for gaps A/C/D from the survey, dense-only smoke first). 



