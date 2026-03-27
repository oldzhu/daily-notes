Yes, there is a real environment mismatch, and it is larger than just gptqmodel.

Direct comparison

- Official:
  - `gptqmodel==5.8.0`
  - `transformers==5.3.0`
  - `torchao==0.16.0`
  - `torch==2.9.1+cu128`
  - `triton==3.5.1`
- fcloud:
  - `gptqmodel==5.7.0`
  - `transformers==4.57.1`
  - `torchao==0.9.0`
  - `triton==3.5.1`

What this means

- `gptqmodel` is different: `5.7.0` vs `5.8.0`
- `transformers` is also different, and this is a much bigger delta: `4.57.1` vs `5.3.0`
- `torchao` is also very different: `0.9.0` vs `0.16.0`
- `triton` matches
- Your `protobuf` check is not meaningful as written because the import name is usually `google.protobuf`, not `protobuf`

So your suspicion is correct, but the mismatch is not only `gptqmodel`. The strongest signal is actually:

1. `transformers 4.57.1 -> 5.3.0`
2. `torchao 0.9.0 -> 0.16.0`
3. `gptqmodel 5.7.0 -> 5.8.0`

Why this matters for the bug

The RoPE/load-path behavior and registration/API behavior are both very plausibly affected by the `transformers` major-version jump. `gptqmodel 5.8.0` may also matter, but I would not isolate the issue to that package alone when `transformers` changed from `4.x` to `5.x`.

What to ask the official owner

Send them the exact delta, not just “was there a change recently”:

```text
Our fcloud environment shows:
- gptqmodel 5.7.0
- transformers 4.57.1
- torchao 0.9.0

Your official evaluation log shows:
- gptqmodel 5.8.0
- transformers 5.3.0
- torchao 0.16.0

Can you confirm whether the base image or dependency resolution changed recently, and whether submissions are expected to run against transformers 5.3.0 / gptqmodel 5.8.0 now?
```

One correction

For protobuf, run this instead:

```bash
python3 - <<'PY'
import json
try:
    import google.protobuf
    print(json.dumps({
        "module": "google.protobuf",
        "import_ok": True,
        "version": getattr(google.protobuf, "__version__", "unknown"),
        "file": getattr(google.protobuf, "__file__", None),
    }, ensure_ascii=False))
except Exception as exc:
    print(json.dumps({
        "module": "google.protobuf",
        "import_ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }, ensure_ascii=False))
PY
```

Recommended next step

Before we change code again, I recommend you do one of these:

1. Ask the official owner with the exact version diff above.
2. Reproduce the official stack in fcloud by upgrading to:
   - `gptqmodel==5.8.0`
   - `transformers==5.3.0`
   - `torchao==0.16.0`

If you want, next I can give you the exact `uv pip install` commands and a minimal isolated reproduction procedure for matching the official environment in fcloud.
