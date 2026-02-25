Good points.

Option 1 (minimum interval):
You’re right—no single value fits all hardware. That’s the main weakness. A fixed floor (like 1 us) is a policy choice, not purely technical. It’s defensible as a software‑mode guardrail, but reviewers may ask “why that number?” unless we tie it to a measurable limit (timer resolution, scheduler tick, or worst‑case latency).

Option 2 (no rearm into the past):
It removes the storm behavior, but if the interval is extremely small, the timer will still fire very frequently in the future. That can still consume significant CPU and could still starve RCU under heavy load—just without the pathological “catch‑up loop.”
So Option 2 reduces the worst case but doesn’t guarantee no RCU stalls if the workload is aggressive enough.

In short:

Option 1 prevents pathological inputs.
Option 2 mitigates a specific failure mode but doesn’t fully cap CPU usage for tiny intervals.
