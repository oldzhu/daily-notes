Where to put important always-follow rules (best practice)

Best for this project: ​.github/copilot-instructions.md
Repository-scoped, persistent, and directly targeted for coding behavior in this codebase.
AGENTS.md
Better for multi-agent/team orchestration conventions, ownership, handoff, and process docs; less direct for strict Copilot coding constraints.
Skills
Better for reusable capabilities/workflows across projects (like a reusable “benchmark runner” skill), not project policy source-of-truth.
Recommended structure

Put hard policy/rules/compliance in ​.github/copilot-instructions.md (source of truth).
Use AGENTS.md only for collaboration/process metadata if needed.
Use skills for reusable procedures, but have them reference project rules rather than replace them.
If you want, I can also add a short “Priority order of instructions” section into ​.github/copilot-instructions.md so this is explicit for future chats.

