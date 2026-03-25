Compacted conversationYou want a diagnostic-only submission variant to force an early preprocess failure and expose the actual module versions from the official environment. I’m checking the latest SOAR submission guidance and the current change-document numbering first, then I’ll give you a proposal only and wait for approval before editing anything.Optimizing tool selection...



Searched for files matching `**/docs/soar_2026_changes/CHANGE_*.{en,zh}.md`, 118 matches

Fetched 2 resources

The toolkit guidance still allows exactly this kind of submission-time diagnostic script through `prepare_model.sh`, and it does not conflict with the official `prepare_env.sh` / `prepare_model.sh --input --output` contract. I’m checking the current demo wrapper now so the proposal can name the precise files and the lowest-risk way to switch to the probe script.

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_model.sh)

**Proposal**

Objective and expected gain

Create a diagnostic-only preprocess entry that prints the actual runtime versions of Python dependencies relevant to the GPTQ failure path, then exits intentionally before any model work begins. Expected gain: hard confirmation of whether the official evaluation environment differs from your fcloud instance on PyTorch, gptqmodel, or transformers, using the official platform’s own last-50-lines failure log.

Rule-compliance check

- I re-checked the latest official toolkit and competition pages today.
- The toolkit still allows custom preprocessing through the submission contract:
  - prepare_model.sh style wrapper is valid under the official prepare_model.sh --input <raw> --output <processed> interface.
  - Submission-time preprocessing and custom code are explicitly allowed.
- This diagnostic change does not modify the base model, does not bypass evaluation rules, does not touch prefix cache or concurrency constraints, and does not attempt to improve scored performance.
- This should be treated as a troubleshooting submission only, not a ranking submission, because it is designed to fail intentionally.

Risk to accuracy and stability

- Accuracy risk: none for the real optimization path, because this would live in a separate diagnostic script and not change the existing preprocess implementation unless we temporarily point the wrapper to it.
- Stability risk: intentional failure by design. That is the point of this probe submission.
- Operational risk: low. The only sensitive point is making sure we fail after printing version lines, so the logs are guaranteed to contain them.

Exact files and functions to change

Primary change:
- Add a new file benchmark/soar/demo_sala/preprocess_model_001.py
  - This new script would:
    - parse --input and --output so it matches the official contract
    - import and print:
      - Python version
      - torch version
      - gptqmodel version
      - transformers version
    - optionally print import file paths for each module if available, which would make environment mismatch diagnosis stronger
    - raise RuntimeError intentionally with a clear marker message after logging

Switching mechanism for submission:
- Option A, lowest ambiguity for official submission:
  - Update prepare_model.sh to call preprocess_model_001.py temporarily
- Option B, lower repo disruption:
  - Keep prepare_model.sh unchanged and you manually swap the called file before packaging
- I recommend Option A for the diagnostic submission because it removes packaging ambiguity.

Documentation required by repo workflow

If approved, I would also create the required bilingual docs for this single troubleshooting feature iteration:

- docs/soar_2026_changes/CHANGE_0054_submission_env_version_probe.en.md
- docs/soar_2026_changes/CHANGE_0054_submission_env_version_probe.zh.md

They would include:
- background and motivation
- compliance statement
- implementation plan
- actual code changes
- validation commands
- expected diagnostic output
- rollback instructions
- next-step suggestions

Validation commands

Local sanity check of the probe script:
- python3 benchmark/soar/demo_sala/preprocess_model_001.py --input /path/to/raw_model --output /tmp/probe_out

If we also repoint the wrapper:
- bash prepare_model.sh --input /path/to/raw_model --output /tmp/probe_out

Expected local result:
- printed version lines for python, torch, gptqmodel, transformers
- immediate intentional RuntimeError

Official confirmation result:
- submission fails early
- official last 50 log lines show those printed versions
- you can compare them directly with your fcloud instance

Success and failure criteria

Success looks like:
- official log contains explicit lines for:
  - python version
  - torch version
  - gptqmodel version
  - transformers version
- and then the intentional failure marker

Failure looks like:
- one of those imports fails before logging all versions
- or the official log truncates before the version lines
- if that happens, I would adjust the probe to print each import result incrementally and flush output aggressively

Recommended implementation shape

I recommend this exact scope for the iteration:
1. Add preprocess_model_001.py as a standalone diagnostic probe
2. Temporarily repoint prepare_model.sh to that file
3. Add the required EN/ZH docs
4. Do not touch the current preprocess_model.py logic

That keeps the existing GPTQ path intact and makes rollback trivial.

Rollback

After you capture the official versions:
1. restore prepare_model.sh to call preprocess_model.py
2. keep preprocess_model_001.py only as a diagnostic artifact, or remove it in a later cleanup iteration

If you approve, I will implement exactly that and keep the scope limited to this diagnostic probe feature.
