Created 7 todos

Step 1 — fix the docs. The wrong claim is in the "Key finding 2 — speed" paragraph and in the "Conclusions" → "Investigate kernel path" bullet, both EN and ZH. 

Multi-Replace String in Files

Replacing 6 lines with 30 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md)


```

```


Replacing 4 lines with 4 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md)


```

```


Replacing 4 lines with 27 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.zh.md)


```

```


Replacing 3 lines with 4 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.zh.md)


```

```


Starting: *Commit doc fix* (2/7)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.zh.md && git commit -m "phase-A: correct kernel-path analysis in CHANGE_0150

Initial doc said 'BF16 dequant fallback' was the cause of long-context slowness.
Code review of modelopt_quant.py shows FP4 tensor cores ARE used on SM120
(flashinfer mm_fp4 + cutlass backend, weights stay in uint8/FP4). Real causes
ranked: (1) quality collapse causes runaway 65k-token generation, (2) per-token
BF16->FP4 quantize overhead, (3) attention still BF16. Queue probes (a,b,c)." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Doc fix committed (`dca76e907`). Now investigation tasks. Confirming approval since these run on fcloud:         

Asked a question (approval)

Starting: *Resume fcloud instance* (3/7)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py console-token-info 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1 | tail -8

Ran terminal command:  cd /home/oldzhu/sglang && for i in 1 2 3 4 5 6 7 8 9 10; do out=$(python3 scripts/fcloud/fcloud_exec.py exec --timeout 15 "echo READY && hostname && date" 2>&1 | tail -3); if echo "$out" | grep -q READY; then echo "READY at attempt $i"; echo "$out"; break; fi; echo "attempt $i: not ready, sleeping 15s"; sleep 15; done

Instance ready. Sync and start server:            

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -10

Compacted conversationProbe (a) is essentially "verify dispatch via runtime" — easier to combine with (b). I'll write a single standalone microbench that:
1. Imports `flashinfer.fp4_quantize` and `flashinfer.mm_fp4` (proves `is_sm120_supported()` returns True at runtime)
2. Times BF16 baseline vs `fp4_quantize+mm_fp4` end-to-end vs `mm_fp4`-only
3. Checks the resulting tensor matches BF16 reference within tolerance (sanity) 

Created [](file:///home/oldzhu/sglang/scripts/fcloud/probe_nvfp4_kernel.py)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py upload --src scripts/fcloud/probe_nvfp4_kernel.py --dst /root/probe_nvfp4_kernel.py 2>&1 | tail -3

Searched for regex `def.*upload|copy.*file|put_file|write_file` (`**/scripts/fcloud/*.py`), 8 results

Ran terminal command:  cd /home/oldzhu/sglang && git add scripts/fcloud/probe_nvfp4_kernel.py && git commit -m "phase-A probe: nvfp4 kernel dispatch + fp4_quantize/mm_fp4 microbench" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -4

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 240 "cd /root/sglang-minicpm && git pull --quiet && cd /root/submission_sim && source ./prepare_env.sh > /tmp/prepare_env.log 2>&1; python3 /root/sglang-minicpm/scripts/fcloud/probe_nvfp4_kernel.py 2>&1 | tail -60" 2>&1 | tail -80

Probe (a) confirmed dispatch: `is_sm120_supported()=True`, FP4 kernels import + execute. Shape bug in my probe — let me inspect what shapes `fp4_quantize` actually returns and fix:     

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 60 "cd /root/submission_sim && source ./prepare_env.sh > /dev/null 2>&1; python3 -c '
import torch
from flashinfer import fp4_quantize, mm_fp4
x = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
w = torch.randn(10880, 4096, dtype=torch.bfloat16, device="cuda")
s = torch.tensor([1.0], dtype=torch.float32, device="cuda")
xf, xs = fp4_quantize(x, 1.0/s)
wf, ws = fp4_quantize(w, 1.0/s)
print("x:", x.shape, x.dtype, "-> xf:", xf.shape, xf.dtype, "xs:", xs.shape, xs.dtype)
print("w:", w.shape, w.dtype, "-> wf:", wf.shape, wf.dtype, "ws:", ws.shape, ws.dtype)
import inspect
print("mm_fp4 sig:", inspect.signature(mm_fp4))
print("mm_fp4 doc head:", (mm_fp4.__doc__ or "")[:400])
'" 2>&1 | tail -30

Replacing 19 lines with 22 lines in [](file:///home/oldzhu/sglang/scripts/fcloud/probe_nvfp4_kernel.py)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add scripts/fcloud/probe_nvfp4_kernel.py && git commit -m "phase-A probe: fix mm_fp4 weight layout (b expects (K/2,N) col-major)" --quiet && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -2

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 240 "cd /root/sglang-minicpm && git pull --quiet && cd /root/submission_sim && source ./prepare_env.sh > /dev/null 2>&1; python3 /root/sglang-minicpm/scripts/fcloud/probe_nvfp4_kernel.py 2>&1 | tail -25" 2>&1 | tail -30

Probes (a)+(b) gave very clean answers. Key results:

| Shape (M×K×N) | BF16 ms | FP4 e2e ms | speedup | quant % |
|---|---|---|---|---|
| decode-1 × 4096×4096 | 0.019 | 0.038 | **0.49×** (slower) | 17% |
| prefill-512 × 4096×10880 | 0.476 | 0.075 | 6.4× | 9% |
| chunk-4096 × 4096×10880 | 2.71 | 0.68 | 4.0× | 2.6% |
| chunk-65536 × 4096×10880 | 41.2 | 10.3 | 4.0× | 5.4% |

Conclusions:
- FP4 cutlass IS used (`is_sm120_supported()=True`, cutlass backend ran)
- ~4× GEMM speedup at prefill/chunk sizes — matches 593/148 TF ratio
- `fp4_quantize` overhead is small (≤10% on relevant shapes)
- Decode-1 is **slower** because memory-bound + 2-kernel launch overhead
- Therefore long-context slowness is **NOT** the kernel — runaway thinking from quality collapse is dominant

Probe (c) (short-context mcq) would add data but needs full server boot; the (a)+(b) result already proves the kernel is healthy. I'll pause instance now and ask before incurring more cost.

Pausing instance + writing findings:     

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && sleep 5 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Instance paused. Now adding probe results to the change doc:         

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md#1-1), lines 1 to 30

Searched for text `## Validation` (`**/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md`), 1 result

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md#100-100), lines 100 to 125

Replacing 4 lines with 39 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md)


```

```


Replacing 4 lines with 36 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.zh.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.en.md docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.zh.md && git commit -m "phase-A: add probe (a)+(b) results to CHANGE_0150

flashinfer fp4_quantize + mm_fp4 cutlass on SM120 confirmed:
- ~4x speedup at prefill/chunk vs BF16 (matches 593/148 TF ratio)
- fp4_quantize <=10% of total at relevant sizes (not a bottleneck)
- decode-1 is 0.49-0.92x BF16 (memory bound + 2-kernel launch)
- => long-context slowness must be runaway thinking, not kernel" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -2

Completed: *Resume fcloud instance* (3/7)

Done. Summary:

**Doc fix (commit `dca76e907`)** — corrected CHANGE_0150 EN/ZH: removed the wrong "BF16 dequant fallback" hypothesis, replaced with source-verified description of the actual kernel chain (`fp4_quantize` → `mm_fp4` cutlass on SM120, weights stay packed FP4) and a ranked list of the real likely causes.

**Probes (a)+(b) (commits `e1ad4eafb`, `660105073`, `2e7a9ed9f`)** — probe_nvfp4_kernel.py ran on fcloud:
- `is_sm120_supported() = True`, flashinfer FP4 path imports + executes
- **~4× speedup** over BF16 at prefill/chunk shapes — matches SM120's 593 TF / 148 TF hardware ratio
- `fp4_quantize` activation cast is ≤10% of total at relevant sizes — **not** the bottleneck we feared
- Decode-1 (M=1) is **0.49–0.92×** BF16 (memory-bound + 2-kernel launch overhead — NVFP4 alone won't speed up single-token decode)
- **Conclusion**: the kernel is healthy; long-context slowness on Phase A had to be runaway thinking from quality collapse, not the GEMM. Phase B (FourOverSix) is the right next step.

**Probe (c) skipped** — (a)+(b) results already prove the kernel is healthy; running mcq with capped `max_tokens` would consume another full server boot for limited extra signal. Left in todo list as optional.

**fcloud state**: paused. 



