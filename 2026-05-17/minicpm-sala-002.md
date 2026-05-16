Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && for i in $(seq 1 30); do out=$(python3 scripts/fcloud/fcloud_exec.py exec "echo ready" 2>&1); if echo "$out" | grep -q "ready"; then echo "READY (attempt $i)"; break; fi; echo "wait $i..."; sleep 15; done

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -20

Now sanity-check the env-default propagation by sourcing on fcloud (no server start), then package.    

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY' 2>&1 | tail -25
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
out = fe.exec_command(b,t,
  "cd /root/submission_sim && bash -c 'unset SOAR_TIER1_LONG_CONTEXT; "
  "grep -n SOAR_TIER1_LONG_CONTEXT prepare_env.sh | head -8; echo ---; "
  "grep -n "chunked-prefill-size" prepare_env.sh'", timeout=60)
print(out)
PY

v21 default-on gate confirmed at lines 16-25 of fcloud's `/root/submission_sim/prepare_env.sh`. Now package and download.    

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY' 2>&1 | tail -30
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
# 1. List submission_sim contents
print(fe.exec_command(b,t,"cd /root/submission_sim && ls -la *.whl *.sh *.py *.jsonl 2>/dev/null && du -sh sglang", timeout=60)[1])
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY' 2>&1 | tail -20
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
out = fe.exec_command(b,t,
  "cd /root/submission_sim && rm -f /root/minicpm_sala_submit_v21.tar.gz && "
  "tar --exclude='__pycache__' --exclude='*.pyc' "
  "-czf /root/minicpm_sala_submit_v21.tar.gz "
  "*.whl *.sh *.py perf_public_set.jsonl sglang && "
  "ls -lh /root/minicpm_sala_submit_v21.tar.gz && "
  "echo 'sha256:' && sha256sum /root/minicpm_sala_submit_v21.tar.gz && "
  "echo 'tar contents (top dirs):' && tar -tzf /root/minicpm_sala_submit_v21.tar.gz | head -20 && "
  "echo '...total entries:' && tar -tzf /root/minicpm_sala_submit_v21.tar.gz | wc -l", timeout=600)
print(out[1])
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY' 2>&1 | tail -10
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
out = fe.exec_command(b,t,"ls -lh /root/minicpm_sala_submit_v21.tar.gz && sha256sum /root/minicpm_sala_submit_v21.tar.gz", timeout=30)
print(out[1])
PY

743MB, well under 2GB cap. Now download to local. Need to use the JupyterLab Files API to download.     

Searched for regex `def download|def fetch|api/contents.*GET|content.*format=base64` (`**/scripts/fcloud/fcloud_exec.py`), no results

Searched for regex `download|tar\.gz` (`**/scripts/fcloud/**`), 13 results

No download helper. JupyterLab's contents API supports GET with `format=base64` for file download. Let me write a small inline helper.     

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_exec.py#100-100), lines 100 to 220

JupyterLab also exposes `/files/<path>` for raw downloads. Let me try that route — much simpler than chunked base64.     

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY' 2>&1 | tail -10
import urllib.request, json
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
# Test HEAD on /files/ endpoint
url = f"{b}/files/minicpm_sala_submit_v21.tar.gz?token={t}"
print("URL:", url[:80] + "...")
req = urllib.request.Request(url, method="HEAD")
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print("status:", r.status)
        print("content-length:", r.headers.get("content-length"))
        print("content-type:", r.headers.get("content-type"))
except Exception as e:
    print("ERR:", e)
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY' 2>&1 | tail -20
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
# List jupyter root
r = fe.api_get(b,t,"/api/contents/")
print("ROOT name:", r.get("name"), "path:", r.get("path"))
print("entries:", [c["name"] for c in r.get("content", [])][:30])
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY' 2>&1 | tail -10
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
# Find where jupyter root really is on disk
out = fe.exec_command(b,t, "ls -la /workspace 2>/dev/null | head -10; echo ---; readlink -f /workspace 2>/dev/null; pwd; ls -la /root/minicpm_sala_submit_v18.tar.gz 2>/dev/null", timeout=30)
print(out[1])
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY' 2>&1 | tail -10
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
out = fe.exec_command(b,t,
  "ln -sf /root/minicpm_sala_submit_v21.tar.gz /workspace/minicpm_sala_submit_v21.tar.gz && "
  "ls -la /workspace/minicpm_sala_submit_v21.tar.gz", timeout=30)
print(out[1])
PY

Ran terminal command:  cd /home/oldzhu/sglang && mkdir -p benchmark/soar/demo_sala && python3 - <<'PY' 2>&1 | tail -20
import urllib.request, sys, time
from scripts.fcloud import fcloud_exec as fe
b,t = fe.load_config()
url = f"{b}/files/minicpm_sala_submit_v21.tar.gz?token={t}"
out_path = "benchmark/soar/demo_sala/minicpm_sala_submit_v21.tar.gz"
t0 = time.time()
with urllib.request.urlopen(url, timeout=600) as r, open(out_path, "wb") as f:
    total = int(r.headers.get("content-length") or 0)
    print(f"size: {total} bytes ({total/1024/1024:.1f} MB)")
    bytes_done = 0
    while True:
        chunk = r.read(4*1024*1024)
        if not chunk:
            break
        f.write(chunk)
        bytes_done += len(chunk)
        if bytes_done % (32*1024*1024) < 4*1024*1024:
            print(f"  {bytes_done/1024/1024:.0f} MB ({bytes_done*100/total:.1f}%)")
print(f"Done in {time.time()-t0:.1f}s -> {out_path}")
PY
ls -lh benchmark/soar/demo_sala/minicpm_sala_submit_v21.tar.gz
sha256sum benchmark/soar/demo_sala/minicpm_sala_submit_v21.tar.gz

**SHA256 matches fcloud (`dddd6f5f...74b75f`).** v21 downloaded intact, 743MB. Pause fcloud now.  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && for i in 1 2 3 4 5; do echo "=== pause attempt $i ==="; out=$(python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1); echo "$out" | tail -3; if echo "$out" | grep -q "OK"; then break; fi; sleep 20; done

**fcloud paused.** Now check Zone.Identifier file (Windows artifact) and add the v21 tarball to .gitignore or just note it. Also need to check if there's also a `.tar.gz:Zone.Identifier` artifact.     

Ran terminal command:  ls -lh /home/oldzhu/sglang/benchmark/soar/demo_sala/*.tar.gz* 2>/dev/null; echo ---; cat /home/oldzhu/sglang/.gitignore 2>/dev/null | head -30; echo ---; grep -E "tar\.gz|submit_v" /home/oldzhu/sglang/.gitignore 2>/dev/null
Completed with input:  ls -lh /home/oldzhu/sglang/benchmark/soar/demo_sala/*.tar.gz* 2>/dev/null; echo ---; cat /home/oldzhu/sglang/.gitignore 2>/dev/null | head -30; echo ---; grep -E "tar\.gz|submit_v" /home/oldzhu/sglang/.gitignore 2>/dev/null

v21 packaging complete. Now drafting **#2-A proposal** (`--torch-compile-max-bs` sweep) so it's ready for your review.     

Searched for text `torch-compile-max-bs` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#180-180), lines 180 to 200

Got the structure. Now draft the #2-A proposal.    

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_torch_compile_max_bs_sweep_20260504.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_torch_compile_max_bs_sweep_20260504.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/PROPOSAL_torch_compile_max_bs_sweep_20260504.en.md docs/soar_2026_changes/PROPOSAL_torch_compile_max_bs_sweep_20260504.zh.md && git commit -m "docs(#2-A): proposal — torch-compile-max-bs sweep on top of v21 (EN+ZH)

Bump --torch-compile-max-bs from 8 to 16 (and optionally 24) via env-gate
SOAR_TORCH_COMPILE_MAX_BS. v21 ships bs=8 by default (graphs cover bs<=8);
Smax tier (max-running-requests=24) currently falls back to eager for
bs in [9,24]. Compiling those buckets should give a measurable Smax decode
gain on the local short-context speed dataset (unlike Tier 1 which only
helped at long context).

Default off (=8 -> v21 byte-equivalent); env-gated; trivial rollback.
Awaiting user approval before any code change or fcloud round." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

## v21 packaged + #2-A proposed

### v21 ready to upload

- **File**: minicpm_sala_submit_v21.tar.gz (743MB)
- **SHA256**: `dddd6f5f8418cd37fde9fdd65d61cdca506244055c0502a78b9475cb1574b75f` (matches fcloud build)
- **Diff vs v20**: one new line in prepare_env.sh defaulting `SOAR_TIER1_LONG_CONTEXT=1` → server now boots with `--chunked-prefill-size 65536 --max-prefill-tokens 65536 --prefill-max-requests 4 --schedule-conservativeness 0.8`. Rollback: `SOAR_TIER1_LONG_CONTEXT=0`.
- **Validated**: Tier1-B run (commit `3f1941f78`) on this exact codepath: acc=78.73% (norm 98.42%, C=0.96), S1/S8/Smax neutral on local.

**Action on you**: upload `minicpm_sala_submit_v21.tar.gz` to https://soar.openbmb.cn manually. After official scoring lands, I'll fetch the leaderboard and update /memories/soar_2026_leaderboard.md.

### fcloud state

- **Paused** (billing released).

### #2-A proposal pushed (commit `bfb2b8b55`)

- PROPOSAL_torch_compile_max_bs_sweep_20260504.en.md / zh
- One-line env gate `SOAR_TORCH_COMPILE_MAX_BS` (default 8 = v21 byte-equivalent); test plan, success/failure matrix, rollback.

**Action on you**: review the proposal. Reply **approve** (or **adjust** with a different bs ceiling) to kick off the patch + fcloud round when convenient. No code changes will be made until you approve.

### Summary of pushes this round

| Commit | Content |
|---|---|
| `a6b16370e` | HAR-based console-config refresh tooling + bilingual howto |
| `3f1941f78` | Tier1-A/B test results + bilingual chat log |
| `edf97175e` | v21 default-on `SOAR_TIER1_LONG_CONTEXT=1` |
| `bfb2b8b55` | #2-A proposal (`--torch-compile-max-bs` sweep) |
    



