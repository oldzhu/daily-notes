Read skill [remote-rocm-jupyter](file:///home/oldzhu/paddle-amd/.github/skills/remote-rocm-jupyter/SKILL.md?vscodeLinkType%3Dskill)

I’m checking the setup/validation docs once so I can answer against this repo’s recorded constraints instead of giving you generic ROCm advice.

Read [](file:///home/oldzhu/paddle-amd/docs/en/setup.md)

Read [](file:///home/oldzhu/paddle-amd/docs/en/validation.md#1-1), lines 1 to 80

Read [](file:///home/oldzhu/paddle-amd/.github/copilot-instructions.md)

Yes. That is a valid working model, and in your situation it is probably the better one.

The key distinction is: WSL does not need to be your authoritative ROCm runtime, but it does need to be a valid ROCm build environment. That matches this repo’s current policy in setup.md and validation.md: use WSL for editing and coordination, and use native Linux ROCm or a remote AMD machine for the real execution verdict.

What is feasible:
- Build a ROCm-enabled Paddle binary locally in WSL.
- Upload that built artifact to 30006 or another remote AMD ROCm instance.
- Run the real tests only on the remote machine.

What must be true for that to work:
- Your WSL distro must have a usable ROCm development toolchain, not just source code.
- At minimum you need things like `hipcc`, ROCm headers/libs, CMake, Ninja, and a Linux userspace compatible with the target build.
- The produced artifact must match the remote runtime well enough:
  - same Python ABI, for example remote 30006 is using Python 3.12 in `/opt/venv`
  - compatible ROCm major/minor, currently the remote image is exposing ROCm 7.2.x
  - correct GPU target coverage, for example `gfx1100`
  - compatible glibc / distro baseline

What does not need to be true:
- You do not need WSL ROCm execution to be trustworthy.
- You do not need to run final BF16 correctness checks locally.
- You do not need local GPU runtime success if the compile and packaging step succeeds.

The practical constraint in your current setup is 30006 networking. Since 30006 cannot resolve GitHub or Gitee, it is a poor place to build from source right now, but it is still a fine place to receive a prebuilt artifact. That means:
- uploading a wheel is good
- uploading a full source tree plus submodules is bad
- uploading a whole build tree is possible but much less clean than a wheel

So the preferred path is:
1. Build a ROCm-enabled wheel locally in WSL, if your WSL has the ROCm SDK/toolchain.
2. Upload the wheel to the remote instance.
3. Install it into `/opt/venv` on the remote instance.
4. Run the real HIP/BF16 validation remotely.

If your WSL does not have a real ROCm dev toolchain, then the answer changes: you can still do all code changes locally, but you should build on a native Linux ROCm box or a better remote instance instead of WSL.

For your current situation, the cleanest recommendation is:
1. Use WSL for code changes and, if possible, wheel build.
2. Treat 30006 as a deploy-and-test target only.
3. Do not rely on 30006’s preloaded Paddle for acceptance, because its creator/cast kernels are already broken.

If you want, I can do the next concrete step and prepare a strict local-build checklist for WSL:
1. verify whether your WSL actually has the ROCm build prerequisites
2. define the exact wheel compatibility target for 30006
3. script the upload/install/test flow for a locally built Paddle wheel
