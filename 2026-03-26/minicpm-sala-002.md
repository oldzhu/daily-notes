You want to stop relying on remote package resolution for the pinned GPTQ stack and switch `prepare_env.sh` to local wheel installs, similar to `flash_attn` and sgl-kernel. I’m keeping this in proposal-only mode per repo policy and will lay out the exact download commands, the script change shape, and the tradeoffs before any edit.**Proposal**

Objective and expected gain

- Download the pinned dependency wheels ahead of time and bundle them with the submission package.
- Update prepare_env.sh to install those local wheels instead of pulling from the network.
- Expected gain:
  - faster and more deterministic `prepare_env.sh`
  - less sensitivity to slow or unstable network
  - no accidental drift to newer `gptqmodel` / `transformers` / `torchao` versions

Rule-compliance check

- This is compliant with the current SOAR submission model.
- `prepare_env.sh` is explicitly allowed to install local packages and custom resources.
- This does not replace the base model or alter evaluation behavior.
- You still need to stay within the 2GB submission limit, so wheel sizes matter.

Risk to accuracy and stability

- Accuracy risk: none directly, because this only freezes installation artifacts for the same versions you already chose.
- Stability risk: low, if the wheels match the target Python/ABI/platform.
- Main operational risk:
  - if a downloaded artifact is not a wheel for `cp310` Linux x86_64, install may fail on official.
  - if a package has no compatible wheel and only sdist is available, this plan needs adjustment.

Exact files to change

- prepare_env.sh
- Required docs for this feature iteration:
  - `docs/soar_2026_changes/CHANGE_0056_prepare_env_local_wheel_installs.en.md`
  - `docs/soar_2026_changes/CHANGE_0056_prepare_env_local_wheel_installs.zh.md`

Recommended local wheel layout

I recommend storing them directly under:
- demo_sala

with filenames like:
- `gptqmodel-5.7.0-...whl`
- `transformers-4.57.1-py3-none-any.whl`
- `torchao-0.9.0-...whl`

This matches your existing pattern for:
- `flash_attn-...whl`
- `sgl_kernel-...whl`

Download commands

Use a dedicated local directory first so you can inspect what pip actually fetched:

```bash
mkdir -p /tmp/soar_demo_wheels
cd /tmp/soar_demo_wheels
```

Download exact wheels only, no sdists:

```bash
python3 -m pip download \
  --only-binary=:all: \
  --no-deps \
  --python-version 310 \
  --implementation cp \
  --abi cp310 \
  --platform manylinux2014_x86_64 \
  gptqmodel==5.7.0 \
  transformers==4.57.1 \
  torchao==0.9.0
```

Then inspect what was downloaded:

```bash
ls -lh /tmp/soar_demo_wheels
```

If you want to move them into the repo after checking:

```bash
cp /tmp/soar_demo_wheels/*.whl /home/oldzhu/sglang/benchmark/soar/demo_sala/
```

If `manylinux2014_x86_64` is too strict for one package, use a simpler fallback on the same target machine:

```bash
python3 -m pip download \
  --only-binary=:all: \
  --no-deps \
  gptqmodel==5.7.0 \
  transformers==4.57.1 \
  torchao==0.9.0
```

What I would change in prepare_env.sh

I would replace the remote installs with local wheel discovery, analogous to your existing flash-attn / sgl-kernel handling.

Planned behavior:

1. define wheel globs for:
   - `gptqmodel-5.7.0-*.whl`
   - `transformers-4.57.1-*.whl`
   - `torchao-0.9.0-*.whl`
2. assert exactly one matching wheel exists for each
3. install them from local files using `uv pip install --force-reinstall --no-deps ...`
4. keep the post-install version probe so the official log still proves the installed versions

Proposed install shape inside prepare_env.sh

```bash
GTPQMODEL_WHEELS=("${SCRIPT_DIR}"/gptqmodel-5.7.0-*.whl)
TRANSFORMERS_WHEELS=("${SCRIPT_DIR}"/transformers-4.57.1-*.whl)
TORCHAO_WHEELS=("${SCRIPT_DIR}"/torchao-0.9.0-*.whl)
```

Then:
- check each array length is exactly 1
- install each wheel locally

Validation commands

After placing the wheels locally:

```bash
bash benchmark/soar/demo_sala/prepare_env.sh
```

Then verify effective versions:

```bash
python3 - <<'PY'
import importlib
import json
for name in ["torch", "gptqmodel", "transformers", "torchao"]:
    module = importlib.import_module(name)
    print(json.dumps({
        "module": name,
        "version": getattr(module, "__version__", "unknown"),
        "file": getattr(module, "__file__", None),
    }, ensure_ascii=False))
PY
```

Success criteria

- `prepare_env.sh` performs no network fetch for those three packages
- official log shows:
  - `gptqmodel==5.7.0`
  - `transformers==4.57.1`
  - `torchao==0.9.0`

Potential issue to watch

- `gptqmodel` may not provide a compatible wheel for your exact target tag in all cases.
- If `pip download --only-binary=:all:` fails for that package, send me the error before we edit the script. In that case, we may need a mixed strategy:
  - local wheel for `transformers` and `torchao`
  - pinned remote install for `gptqmodel`

If you approve, I will:
1. update prepare_env.sh to use local wheels
2. add the required `CHANGE_0056` EN/ZH docs
3. keep the change scoped only to this local-wheel installation feature
