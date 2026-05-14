Starting: *Add in-place mcq patch toggle helper* (1/3)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1040-1040), lines 1040 to 1140

Now creating the in-place toggle helper: 

Created [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/toggle_mcq_thinking_patch.py)

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY'
import json, shutil, tempfile, subprocess, sys
from pathlib import Path

# Build a tiny fake model dir with a sample chat_template
tmp = Path(tempfile.mkdtemp())
tok = tmp / "tokenizer_config.json"
orig = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
tok.write_text(json.dumps({"chat_template": orig}, ensure_ascii=False, indent=2))

helper = "benchmark/soar/demo_sala/toggle_mcq_thinking_patch.py"

def run(mode):
    r = subprocess.run([sys.executable, helper, "--model-dir", str(tmp), "--mode", mode], capture_output=True, text=True)
    print(f"--- {mode} ---")
    print(r.stdout.strip())
    if r.returncode: print("STDERR:", r.stderr)

run("status")
run("on")
run("status")
run("on")  # idempotent
run("off")
run("status")
run("off")  # idempotent

# Verify content is back to original after off
final = json.loads(tok.read_text())["chat_template"]
print("ROUNDTRIP_OK:", final == orig)
shutil.rmtree(tmp)
PY


Helper works (on/off/idempotent/roundtrip all green). Now researching the fcloud start/stop API.  

Starting: *Research fcloud start/stop API* (2/3)

Read [](file:///home/oldzhu/sglang/scripts/fcloud)



Fetched https://console.cnomnibot.com/omnibot

Ran terminal command:  cat ~/.fcloud_config 2>/dev/null | sed 's/TOKEN=.*/TOKEN=<redacted>/' ; echo '---' ; head -60 /home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py



Fetched 3 resources



Fetched 2 resources



Fetched 3 resources



Fetched 3 resources

Ran terminal command:  cd /home/oldzhu/sglang && git add benchmark/soar/demo_sala/toggle_mcq_thinking_patch.py && git status --short && git commit -m "CHANGE_0140: add in-place toggle helper for fcloud A/B testing

toggle_mcq_thinking_patch.py reuses CHAT_TEMPLATE_MCQ_PATCH_PREAMBLE/MARKER
from preprocess_model.py to apply or revert the Jinja preamble directly on
an already-quantized model dir's tokenizer_config.json. Avoids re-running
preprocess_model.py / re-quantization between A/B variants on fcloud.

Production submission flow (preprocess_model.py) is unchanged.

Tested locally: status/on/off/idempotent/roundtrip all green." 2>&1 | tail -5 && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -2

Completed: *Research fcloud start/stop API* (2/3)

## Summary

### 1. Toggle helper — DONE (commit `d65ffe0d7`)

toggle_mcq_thinking_patch.py — imports `CHAT_TEMPLATE_MCQ_PATCH_PREAMBLE`/`MARKER` from preprocess_model.py so the patched template is byte-identical to the submission path.

```bash
# Usage on fcloud (no re-quantization needed)
python3 toggle_mcq_thinking_patch.py --model-dir /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 --mode status
python3 toggle_mcq_thinking_patch.py --model-dir /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 --mode on    # patch
python3 toggle_mcq_thinking_patch.py --model-dir /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 --mode off   # revert
```

Local test: status / on / on (idempotent noop) / off / off (idempotent noop) / roundtrip-equality all green.

### 2. Fcloud start/stop API research — findings

**Result: there is NO public API documented for instance start/stop.** Verified by reading:
- `docs.cnomnibot.com/zh/omnibot/userguide/tarin/tarinservice.html` (训练服务 — task list) — describes the UI start/stop buttons but exposes no API.
- `docs.cnomnibot.com/zh/cloudService/gpu/userguide/instance.html` and `instanceLifeCycle.html` — describes states ("运行中"/"关机"/"正在开机") but no REST endpoint.
- `docs.cnomnibot.com` site-wide — no SDK / CLI / OpenAPI spec for console operations. The "API 密钥" referenced is for **model API service** (inference), not console actions.
- `console.cnomnibot.com/api/docs` → HTTP 404.

**Why our `shutdown` works without an API key**: it shuts down the **JupyterLab server** running inside the instance (via `FCLOUD_URL` = `http://223.167.85.181:12369`, the JupyterLab port). That kills the kernel/JupyterLab process — the omnibot platform then bills/transitions the task accordingly. But **starting** an instance must come from outside (the JupyterLab port doesn't exist until the instance is up), so it requires the console's own internal endpoint.

**Recommended path forward (need your help to capture)**:

The console SPA's start button hits an internal endpoint we can call directly — but we need to capture it. Easiest path:

1. Open https://console.cnomnibot.com/omnibot in Chrome/Edge, log in.
2. Press **F12** → **Network** tab → check "Preserve log".
3. Click the green ▶ start button on the stopped instance.
4. Find the POST/PUT request (likely under a path like `/api/.../task/.../start` or `/api/training/.../start`).
5. Right-click → **Copy → Copy as cURL** (or Copy as fetch).
6. Paste it back to me (mask any obvious secrets if you want; I'll need: full URL, method, request headers including `Authorization`/`Cookie`, and request body).

Also do the same on the ⏸ stop button so we have both.

Once I have those two cURL snippets I can:
- Add a `~/.fcloud_console_config` for the auth token/cookie (separate from the JupyterLab `~/.fcloud_config`).
- Wire `python3 fcloud_workflow.py start` / `stop` into the workflow CLI.
- Add a `start-and-wait` that polls until JupyterLab is reachable, then runs `setup` if needed.

I'll hold on the wiring (todo #3) until you provide the captured request — implementing it speculatively against an unknown endpoint shape would be guesswork. 



