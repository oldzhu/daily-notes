Added the pros/cons + regression-impact notes for the 3 fix options (A/B/C) directly into the RCA doc and pushed it.

Updated notes section: 2026-02-04-f8850bc3986562f79619-taprio-rcu-stall-rca.md (see “Pros/cons & regression risk (Options A/B/C)”)
Pushed commit: b0e8c1f on origin/main
Key takeaways captured in the doc:

Option A: simplest/strongest mitigation but rejects sub-us software schedules.
Option B: reduces hard-IRQ starvation risk but may add timing jitter.
Option C: most correct long-term but highest complexity and may intentionally skip missed transitions under overload.

