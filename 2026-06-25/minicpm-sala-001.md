I’m validating the refreshed console auth and bringing the fcloud instance up; if that succeeds, I’ll run the smallest remote kernel reproducer first and pause the instance again after collecting the output.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py console-token-info

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance

The instance resumed cleanly. Next I’m checking remote shell access and then running the isolate reproducer against the live `libw4a8_fused_gemm.so` in the submission environment.

Starting: *Run focused remote diagnostics* (2/4)

This batch verifies the remote shell is reachable and runs the minimal isolate test in the same remote runtime that serves the model, so the result reflects the actual live kernel/library path.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'hostname && nvidia-smi --query-gpu=name --format=csv,noheader'

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/submission_sim && source ./prepare_env.sh >/tmp/prepare_env.log 2>&1 && cd /root/sglang-minicpm && python3 benchmark/soar/demo_sala/test_isolate.py'

The live reproducer matches the earlier split exactly: both `ones`-activation cases pass, both random-activation cases fail with large errors. I’m pulling the existing targeted debug tests next so we can see whether they already isolate A staging, B staging, or K-order more directly.

Searched for files matching `sglang` · `benchmark/soar/demo_sala/test_*.py`, 12 matches

Searched for regex `verify_a|compare_a|weight_diag|minimal|diag_w4a8|k_range|m_range|n_range` (`**/benchmark/soar/demo_sala/test_*.py`), 1 result

This batch reads the most targeted existing diagnostics so I can choose one remote check that sharpens the hypothesis instead of spraying broad tests.

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/test_minimal.py)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/test_compare_a.py)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/test_verify_a.py)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/test_diag_w4a8.py)

The existing remote isolate test is confirmed. I’m now running one sharper diagnostic: a one-hot and two-hot activation probe against the live kernel, which will tell us whether K indices are being selected/accumulated incorrectly rather than just “random activations fail.”

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/submission_sim && source ./prepare_env.sh >/tmp/prepare_env.log 2>&1 && cd /root/sglang-minicpm && python3 benchmark/soar/demo_sala/test_diag_w4a8.py'

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "$(cat <<'EOF'
cd /root/submission_sim
source ./prepare_env.sh >/tmp/prepare_env.log 2>&1
cd /root/sglang-minicpm
python3 - <<'PY'
import torch

torch.ops.load_library('/root/submission_sim/libw4a8_fused_gemm.so')
K, N, M, g = 128, 8, 128, 128
qw = torch.zeros(K//8, N, dtype=torch.int32, device='cuda')
for k in range(K):
    for n in range(N):
        kp = k // 8
        kbit = (k % 8) * 4
        val = (k * N + n) % 16
        qw[kp, n] |= (val << kbit)
qz = torch.zeros((K//g, N//8), dtype=torch.int32, device='cuda')
sc = torch.ones((K//g, N), dtype=torch.bfloat16, device='cuda')

w_ref = torch.zeros(K, N, dtype=torch.float32, device='cuda')
for k in range(K):
    for n in range(N):
        kp = k // 8
        kbit = (k % 8) * 4
        w4 = (qw[kp, n].item() >> kbit) & 0xF
        w_ref[k, n] = float(w4) - 1.0
w_fp8 = w_ref.to(torch.float8_e4m3fn)

# one-hot probe
A = torch.zeros(M, K, dtype=torch.bfloat16, device='cuda')
A[0, 5] = 1
A[1, 13] = 1
A[8, 21] = 1
A[9, 29] = 1
a_fp8 = A.to(torch.float8_e4m3fn)
r = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw, qz, sc, a_fp8, N, K, g)
ref = (a_fp8.to(torch.float32) @ w_fp8.to(torch.float32)).to(torch.bfloat16)
print('=== one-hot probe ===')
for row in [0, 1, 8, 9]:
    print('row', row, 'kernel', [float(x) for x in r[row, :].cpu()], 'ref', [float(x) for x in ref[row, :].cpu()])
print('one-hot allclose', torch.allclose(r, ref, atol=1.0), 'maxdiff', float((r-ref).abs().max().item()))

# two-hot weighted probe
a2 = torch.zeros(M, K, dtype=torch.bfloat16, device='cuda')
a2[0, 3] = 1
a2[0, 19] = 2
a2_fp8 = a2.to(torch.float8_e4m3fn)
r2 = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw, qz, sc, a2_fp8, N, K, g)
ref2 = (a2_fp8.to(torch.float32) @ w_fp8.to(torch.float32)).to(torch.bfloat16)
print('=== two-hot probe ===')
print('row 0 kernel', [float(x) for x in r2[0, :].cpu()])
print('row 0 ref   ', [float(x) for x in ref2[0, :].cpu()])
print('two-hot allclose', torch.allclose(r2, ref2, atol=1.0), 'maxdiff', float((r2-ref2).abs().max().item()))
PY
EOF
)"

The random-activation diagnostic is decisive: almost every output is wrong, not just a corner case. The ad hoc one-hot probe command got mangled by shell quoting, so I’m switching to a cleaner path and re-running that probe in a controlled way.

Searched for regex `def upload_file|class Fcloud|upload_file\(` (`**/scripts/fcloud/**/*.py`), 4 results

Checked terminal output

I found a direct upload helper in fcloud_exec.py, which gives us a reliable way to push a tiny probe script to the instance without more shell-quoting issues.

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_exec.py#210-210), lines 210 to 280

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_97eolXh07KPHwRr3ENw8ZGJw__vscode-1779749830455/content.txt#1-1), lines 1 to 220

I’ve got a reliable upload path now. I’m using it to push a tiny basis-probe script to the instance so we can test K-index selection directly without more shell quoting noise.

Created [](file:///tmp/w4a8_basis_probe_20260526.py)

Searched for regex `def load_config|FCLOUD_URL|FCLOUD_TOKEN|read_config` (`**/scripts/fcloud/fcloud_exec.py`), 13 results

The probe script is ready locally. I’m uploading it with the built-in helper, then executing it remotely inside the submission environment so we get a clean K-selection readout.

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY'
from scripts.fcloud import fcloud_exec
url, token = fcloud_exec.load_config()
ok = fcloud_exec.upload_file(url, token, '/tmp/w4a8_basis_probe_20260526.py', '/root/w4a8_basis_probe_20260526.py', chunk_size_mb=1)
print('upload_ok', ok)
PY

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_exec.py#60-60), lines 60 to 105

The upload succeeded. I’m executing the probe remotely now; this should tell us whether one-hot K selection is already wrong, or whether the failure only appears once multiple K terms are accumulated.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/submission_sim && source ./prepare_env.sh >/tmp/prepare_env.log 2>&1 && python3 /root/w4a8_basis_probe_20260526.py'

Killed terminal

The first basis probe exposed something important, but the weight pattern accidentally aliased those selected K rows, so the one-hot part was weaker than it looked. I’m rerunning it with a K-distinct weight pattern to separate “K selection is wrong” from “multi-K accumulation is wrong.”

Created [](file:///tmp/w4a8_basis_probe_distinct_20260526.py)

I’m uploading and executing the distinct-pattern probe now. This should tell us whether single-K selection is actually correct and whether the breakage starts specifically at multi-term accumulation.

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY'
from scripts.fcloud import fcloud_exec
url, token = fcloud_exec.load_config()
ok = fcloud_exec.upload_file(url, token, '/tmp/w4a8_basis_probe_distinct_20260526.py', '/root/w4a8_basis_probe_distinct_20260526.py', chunk_size_mb=1)
print('upload_ok', ok)
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/submission_sim && source ./prepare_env.sh >/tmp/prepare_env.log 2>&1 && python3 /root/w4a8_basis_probe_distinct_20260526.py'

The distinct probe changed the diagnosis: this is no longer best explained as a B/K mapping problem. The output matches a row-folding pattern, so I’m running one final direct check for that specific hypothesis before I report.

Created [](file:///tmp/w4a8_rowfold_probe_20260526.py)

I’m uploading and running the row-fold probe now. If the hypothesis is right, a nonzero input only on row 8 will show up on row 0 in the kernel output.

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY'
from scripts.fcloud import fcloud_exec
url, token = fcloud_exec.load_config()
ok = fcloud_exec.upload_file(url, token, '/tmp/w4a8_rowfold_probe_20260526.py', '/root/w4a8_rowfold_probe_20260526.py', chunk_size_mb=1)
print('upload_ok', ok)
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/submission_sim && source ./prepare_env.sh >/tmp/prepare_env.log 2>&1 && python3 /root/w4a8_rowfold_probe_20260526.py'

The row-fold hypothesis is confirmed. I’ve collected the needed outputs, and I’m pausing the fcloud instance now before I summarize the diagnosis and what it means in the kernel.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance

Starting: *Pause fcloud instance* (4/4)

The first pause attempt hit the known transient 504 path. I’m retrying once now; this is expected behavior for the console API.

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance

Updated todo list

I have the full live diagnosis now and the instance is paused. I’m marking the last step complete and then I’ll summarize the exact failure pattern and the kernel surface it implicates.

Completed: *Pause fcloud instance* (4/4)

I ran the diagnosis on fcloud, confirmed the live behavior, and paused the instance again afterward.

The main finding is that the failure is not best explained by B/K-order packing anymore. The live kernel is folding the lower 8 rows of each `m16` tile into the upper 8 rows. The evidence is:

1. The original remote isolate reproducer still shows the same split:
   - simple weights + ones activations: pass, `maxdiff=0`
   - simple weights + random activations: fail, `maxdiff=422`
   - random weights + ones activations: pass, `maxdiff=0`
   - random weights + random activations: fail, `maxdiff=442`

2. The random-activation diagnostic is broad failure, not a corner case:
   - `32421 / 32768` outputs differ from FP8 ref by more than `1.0`
   - mean absolute error is `76.5`

3. The decisive probe was the distinct one-hot row test. With different rows activating different K positions, the kernel produced:
   - row 0 = ref row 0 + ref row 8
   - row 1 = ref row 1 + ref row 9
   - rows 8 and 9 = all zeros

4. The direct row-fold probe confirmed it cleanly:
   - input only had nonzero activation on row 8
   - kernel output row 0 = ref row 8
   - kernel output row 8 = 0

That means the bug is in the A-side row mapping for the `m16n8k32` MMA path, not just “random activations are hard” and not primarily a B fragment issue. In the current kernel, the suspect surface is the A fragment packing at w4a8_fp8_qmma.cu, where `a_regs[2]` and `a_regs[3]` are assumed to represent the independent lower-half rows `row0 + 8`. The row-fold result shows those lower-half contributions are being accumulated into the upper-half outputs instead. The epilogue at w4a8_fp8_qmma.cu is not the primary problem, because the wrong sum already exists before store.

The B path at w4a8_fp8_qmma.cu is no longer the top suspect. A useful counterexample is the two-hot single-row probe: once only row 0 is active, two-K accumulation matched reference exactly. So multi-K accumulation can work when there is no cross-row interaction. What breaks is the separation between row `r` and row `r + 8`.

No repo files were changed in this round. The fcloud instance is paused.

Natural next steps are:

1. Rework the A fragment packing in w4a8_fp8_qmma.cu and validate first with the row-fold probe.
2. Keep the B fragment path unchanged for the first fix attempt, since the new evidence points much more strongly at A-row aliasing than at B/K permutation.

