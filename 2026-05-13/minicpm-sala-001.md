Ran terminal command:  cd /home/oldzhu/sglang/benchmark/soar/demo_sala && ls -la *.tar.gz 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang/benchmark/soar/demo_sala && mkdir -p /tmp/v18l /tmp/v20l && tar -tzvf minicpm_sala_submit_v18.tar.gz > /tmp/v18l/list.txt 2>&1 && tar -tzvf minicpm_sala_submit_v20.tar.gz > /tmp/v20l/list.txt 2>&1 && echo "v18 entries: $(wc -l < /tmp/v18l/list.txt)" && echo "v20 entries: $(wc -l < /tmp/v20l/list.txt)"

Ran terminal command:  awk '{print $NF}' /tmp/v18l/list.txt | sort > /tmp/v18l/paths.txt && awk '{print $NF}' /tmp/v20l/list.txt | sort > /tmp/v20l/paths.txt && echo "=== Only in v20 (added) ===" && comm -13 /tmp/v18l/paths.txt /tmp/v20l/paths.txt && echo "=== Only in v18 (removed in v20) ===" && comm -23 /tmp/v18l/paths.txt /tmp/v20l/paths.txt

Ran terminal command:  echo "=== v20 added file sizes (bytes) ===" && grep -E "(\.v19orig|fp8_blockwise\.py|utils_w4a8_fp8\.py|minicpm_eagle3\.py)" /tmp/v20l/list.txt && echo "" && echo "=== Total uncompressed size v18 vs v20 ===" && awk '{s+=$3} END{print "v18:", s}' /tmp/v18l/list.txt && awk '{s+=$3} END{print "v20:", s}' /tmp/v20l/list.txt

Ran terminal command:  awk '{print $3, $NF}' /tmp/v18l/list.txt | sort -k2 > /tmp/v18l/sz.txt && awk '{print $3, $NF}' /tmp/v20l/list.txt | sort -k2 > /tmp/v20l/sz.txt && join -j 2 /tmp/v18l/sz.txt /tmp/v20l/sz.txt | awk '$2!=$3 {print $3-$2, $1}' | sort -n | tail -15

Ran terminal command:  grep "sgl_kernel-0.3.20.post1" /tmp/v18l/list.txt /tmp/v20l/list.txt

Found it. **The +34.5 MB delta is essentially 100% from the rebuilt `sgl_kernel-0.3.20.post1` wheel** (515.6 MB → 550.0 MB, +32.9 MB). All other diffs are negligible:

| Source of growth | Bytes |
|---|---|
| `sgl_kernel-0.3.20.post1-cp310-abi3-linux_x86_64.whl` | **+34,459,858** (≈ 32.9 MB) |
| `prepare_env.sh` (v20 flags + comments) | +8,600 |
| `sglang/.../gptq.py` | +4,350 |
| `sglang/.../minicpm_backend.py` | +3,645 |
| 7 newly-added Python files (eagle3 stub, w4a8/fp8_blockwise, 4× `.v19orig` backups) | +145 KB total |
| Misc tuning JSON config nudges | +~30 KB |
| **Total** | **+34.6 MB** ≈ tarball delta (+34.5 MB) ✓ |

### Why the wheel grew

The wheel on fcloud (`/root/submission_sim/sgl_kernel-0.3.20.post1-...whl`, dated 2026-04-28) is the rebuild produced after the SM120 Marlin / NVFP4 / W4A8 kernel work landed in sgl-kernel since v18 (March 30 build). All those new `.cu` translation units (mxfp8 QMMA paths, additional Marlin tile sizes for SM120, NVFP4 KV scaffolding, FP8 blockwise GEMM probes) compile into extra `.o` objects that the linker bundles into the single shared library inside the wheel.

### Is this a problem?

- **2 GB submission limit**: total package is 778 MB → still ≈ 1.25 GB headroom. No issue.
- **Functional**: the rebuilt wheel is what we benchmarked (Test 12, 13f-1 etc.), so it is the version we want to submit.
- **Optional shrink** (only if you want to claw back the 33 MB):
  1. Strip debug symbols: `strip --strip-unneeded` on the `.so` inside the wheel (often −20–40% on large CUDA libs).
  2. Re-link with `-Wl,--gc-sections` and disable unused kernels (we currently compile lots of paths we never dispatch on SM120, e.g. SM90/SM100 Marlin variants, FP8 blockwise probes that aren't on the hot path).
  3. `xz` instead of `gzip` for the outer tar: `tar --exclude='__pycache__' -cJf …tar.xz` typically saves another 5–10% on top.

Recommendation: **leave v20 as-is**. The 33 MB is real new functionality (SM120 Marlin tiles + NVFP4 scaffolding) that we may want to dispatch into in Round 14+. We are nowhere near the 2 GB limit, and stripping/relinking the wheel is a separate optimization with its own risk surface.Continued with Autopilot: Task not yet completeInvestigated v20 vs v18 size delta: the +34.5 MB increase is ~100% explained by the rebuilt `sgl_kernel-0.3.20.post1` wheel (515.6 MB → 550.0 MB, +32.9 MB) which now contains the SM120 Marlin / NVFP4 / W4A8 kernel additions made since the March 30 v18 build. All other file diffs (prepare_env.sh, gptq.py, minicpm_backend.py, 7 new small Python files) total <200 KB. Submission is still ~1.25 GB under the 2 GB limit, so no action recommended. 


