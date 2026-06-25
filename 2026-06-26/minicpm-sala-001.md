I’m turning the vague “investigate timeout” item into concrete slices so the next fix targets the actual failure mode rather than rerunning the full harness blind.

Added 2 todos

The next fix is not another full accuracy rerun. The next fix is to isolate which exact sample class starts failing after about 76 and turns into the 92/150 timeout wall.

Right now the evidence in TEST_RESULTS_TRACKING.md and CHANGE_W4A8_001_iteration_003.en.md points to this sequence:
1. The QMMA row-fold bug is fixed.
2. The server can answer simple prompts correctly.
3. The full harness later stalls on a few long-context requests, then the wrapper hits 3600s.

So the correct next move is:

1. Identify the exact samples around the slowdown window.
Use the partial output and server logs to map sample indices around 76–92 to task type, prompt length, and whether output length is exploding. The key question is: are we seeing true compute slowness, or runaway generation that keeps the request alive until the client read timeout?

2. Reproduce only those samples on fcloud.
Run a tiny subset, ideally 1 to 5 samples, one at a time. That gives a cheap discriminating check:
- If a single sample reproduces the timeout, the issue is request-specific.
- If the same sample finishes alone but times out in sequence, the issue is state buildup or scheduling.

3. Only then decide the fix slice.
The likely branches are:
- Runaway generation / thinking does not terminate on certain long tasks.
- Very long prefill or decode path regressed only on specific long contexts.
- A server-side scheduling or graph-capture interaction appears only after enough requests.

My current best hypothesis is runaway long-context generation on a small set of samples, not a kernel math bug. The fastest disconfirming check is a single-sample reproduction with prompt length, output length, and wall time captured.

If you want, the next concrete action is for me to start fcloud again and run that targeted subset investigation. That needs your approval because it uses the shared fcloud instance.
