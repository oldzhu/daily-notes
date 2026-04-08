Read skill [agent-customization](file:///home/oldzhu/.vscode-server/extensions/github.copilot-chat-0.42.3/assets/prompts/skills/agent-customization/SKILL.md?vscodeLinkType%3Dskill)

I’m gathering the task brief from the hackathon page and checking the workspace state so I can draft a concrete delivery plan, documentation structure, and project setup approach.

Inspect workspace



Read [](file:///home/oldzhu/paddle-amd)

Fetched https://github.com/PaddlePaddle/community/blob/25db189affe6666252051a9e6857facff9842e46/hackathon/hackathon_10th/【Hackathon_10th】文心合作伙伴任务合集.md

Read [](file:///home/oldzhu/paddle-amd)

The task brief is clear and the workspace is clean. I’m now pulling the likely upstream code touchpoints in Paddle and the current PaddleX workaround so the plan can name concrete files, symbols, and verification gates instead of staying generic.



Fetched https://github.com/PaddlePaddle/PaddleX/compare/release/3.3...vivienfanghuagood:PaddleX:dev_rocm70

Searched paddlepaddle/paddle for "HIP BF16 ROCm bfloat16 operator registration MIOpen conv BF16 phi backends gpu rocm data type support. Find relevant files/symbols for adapting HIP BF16 precision type in Paddle, especially type definitions, kernel registration, MIOpen convolution, and existing CUDA BF16 patterns.", 64 results

The likely Paddle touchpoints are now concrete: ROCm BF16 type plumbing already exists in parts of PHI and rocBLAS/MIOpen helpers, but HIP GPUDNN conv registration and some HIP-only type gates still exclude BF16, which matches the PaddleX workaround. I need three scope decisions from you before I lock the repo structure and execution plan.

Analyzing your answers...

I’ve got the critical scope decisions: this will be a coordination repo with shared project instructions, and validation has to assume split environments between WSL development and native/remote ROCm testing. I’m checking existing memory entries before saving the project plan, then I’ll present the plan in a form you can approve directly.

Read memory

I’m writing the plan into session memory now, with explicit phases, verification, and a documentation system that forces bilingual tracking at each material decision.

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/b486c99776a8a4408935da2c9ddf21ab/GitHub.copilot-chat/memory-tool/memories/NzczN2M0ZjQtNGI4Yi00ZTUwLWE5MjQtNTU0OTBiYjc0NDc2/plan.md)

**Task Review**
This AMD task is a strong choice because it targets a real upstream gap with clear acceptance criteria and measurable impact.

The key requirements are:
1. Fix Paddle itself so HIP BF16 works on ROCm, with PaddleOCR-VL-1.5 able to run end-to-end in BF16 on AMD GPU and produce correct results.
2. Submit an Issue and PR to Paddle develop.
3. Submit an Issue and PR to PaddleX develop to remove the current ROCm BF16 workaround.
4. Include tests plus AMD-GPU validation screenshots/results.

The technical signal is good: this is not a greenfield feature. Paddle already has partial ROCm BF16 plumbing in helper/type layers, while PaddleX currently carries explicit workarounds for ROCm BF16, including BF16 disablement, forcing visual modules to FP32, and deleting conv-related fuse passes. That strongly suggests a narrow upstreamable root-cause fix is possible, most likely around HIP BF16 dispatch and GPUDNN/MIOpen conv-related support rather than a full-stack rewrite.

**Recommended Plan**
## Plan: HIP BF16 for Paddle on AMD

Use this workspace as a coordination repo, not the Paddle source tree. The recommended path is: bootstrap a bilingual, documentation-first tracking repo here; develop against separate Paddle and PaddleX clones; reproduce the ROCm BF16 failure on a native Linux ROCm host; implement the smallest upstreamable Paddle changes to enable HIP BF16 conv-related execution; then remove PaddleX workarounds and submit the required Issue/PR set with verification evidence.

**Steps**
1. Phase 1: Initialize a git repo in this folder and turn it into the project control plane, not the main Paddle fork.:
2. Phase 1: Create a shared workspace instruction file at .github/copilot-instructions.md that enforces your process rules: bilingual docs for every important step and decision, English/Chinese counterpart links at the top of each doc, issue/PR evidence tracking, and disciplined change logging.
3. Phase 1: Create paired Chinese and English documents for plan, design, dev log, decision log, validation log, and change log. Keep section structures mirrored so updates stay synchronized.
4. Phase 1: Keep Paddle and PaddleX clones outside this repo to avoid nested-git problems. Store only docs, scripts, exported patches, screenshots, issue drafts, and milestone tracking here.
5. Phase 2: Clone Paddle develop and PaddleX develop separately and reproduce the BF16 failure path for PaddleOCR-VL-1.5 on a native Linux ROCm machine. This is the main blocker for implementation.
6. Phase 2: Confirm exactly what breaks when PaddleX workarounds are removed, especially the BF16 availability gate, visual FP32 pinning, and deleted conv fuse passes. Record failing operators, stack traces, and environment details in both languages.
7. Phase 3: Implement the smallest Paddle root-cause fix first. Current high-probability touchpoints are HIP-only type gating plus GPUDNN/MIOpen conv and conv-transpose BF16 registration/support.
8. Phase 3: Use existing CUDA BF16 support only as a structural reference. Validate every HIP change against the actual ROCm and MIOpen APIs available on the target machine.
9. Phase 3: Add focused regression tests around the enabled BF16 path, starting at operator level, then an integration smoke test.
10. Phase 4: After Paddle BF16 works, prepare the PaddleX cleanup PR to remove temporary BF16 disablement and FP32 fallback behavior, while keeping any still-independent ROCm limitations explicitly documented.
11. Phase 5: Prepare the submission package: Paddle Issue, Paddle PR, PaddleX Issue, PaddleX PR, validation screenshots, environment capture, and a concise FP32 vs BF16 comparison for memory and runtime.
12. In parallel after Phase 1: issue-template drafting, repro-script writing, benchmark-script writing, and bilingual documentation setup can proceed without waiting for code changes.

**Important Decisions**
1. This repo should be a coordination repo, not a direct Paddle fork.
2. The bilingual documentation rule should be project-wide and enforced through workspace-shared copilot instructions.
3. WSL should be treated as editing/orchestration only unless proven stable for AMD GPU work.
4. Final validation should target a native Linux ROCm machine, preferably remote if local Windows/WSL remains unreliable.

**Winning Strategy**
1. Keep the first Paddle patch narrow and reviewable. A minimal root-cause fix with clean tests and strong evidence is more likely to win than a large ROCm patch bundle.
2. Document everything in both languages as you go, not afterward. That will help both hackathon judging and upstream review.
3. Separate clearly what belongs in Paddle and what belongs in PaddleX. The best story is: Paddle gets the real fix; PaddleX removes temporary workaround code.

I saved this plan into session memory so it is ready for handoff. I did not initialize the git repo yet, because I am currently restricted to planning only. If you approve this plan, the first implementation step should be to initialize the repo here and create the bilingual documentation skeleton plus .github/copilot-instructions.md.
