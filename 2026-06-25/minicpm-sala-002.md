I’m turning the diagnosis into a concrete, narrow fix proposal first, because this repo requires proposal-plus-approval before any kernel edit.

I’m checking the SM120 hardware note before I lock the proposal, so the plan stays aligned with the repo’s kernel-change rules and with the actual MMA contract on this GPU.

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md)

I checked the SM120 hardware note in SM120_RTX_PRO_HARDWARE.md. It is consistent with our path: warp-level `mma.sync` on SM120 is the right primitive, so this is a correctness fix inside the existing warp-level FP8 MMA path, not a redesign.

**Proposal**
1. Objective and expected gain:
Fix the row-fold bug in w4a8_fp8_qmma.cu, where rows `8..15` of each `m16` tile are being accumulated into rows `0..7`. Expected gain is correctness recovery, not speed: the narrow success criterion is that the row-fold probe passes and test_isolate.py stops failing on random activations. If correct, this should move end-to-end accuracy from effectively unusable toward the prior dense GPTQ+FP8-KV baseline instead of `0%`.

2. Falsifiable local hypothesis:
The A fragment packing at w4a8_fp8_qmma.cu through w4a8_fp8_qmma.cu does not match the actual SM120 `m16n8k32.row.col` lane contract. The lower-half rows are not being represented as independent `row+8` outputs. That matches the live probe result where an activation only on row 8 produces output on row 0 and leaves row 8 zero.

3. Exact files/functions to change:
Source:
w4a8_fp8_qmma.cu
Function:
`w4a8_fp8_qmma_kernel`
Edit slice:
the A-fragment load block around w4a8_fp8_qmma.cu
Likely unchanged in the first attempt:
the B-fragment load block around w4a8_fp8_qmma.cu
Validation helpers:
either add or formalize the row-fold and basis probes under demo_sala
Required docs for this feature if approved:
`CHANGE_W4A8_001_iteration_003.en.md` and `.zh.md`
matching chat log under chat

4. Rule-compliance check:
This stays within the allowed submission path.
No eval harness changes.
No prepare-env launch bypass.
No switch away from the current baseline config.
No load-time full INT4→FP8 conversion in HBM.
Only submission-side kernel behavior changes.

5. Risk to accuracy and stability:
Main risk is that my first A-fragment remap is still not the exact PTX contract, in which case the row-fold bug may change shape rather than disappear.
Low risk of speed regression from the first edit because I plan to keep the same kernel structure and only change fragment packing.
Low risk to unrelated paths because the change is isolated to this kernel.

6. Validation plan:
First focused check after the first edit:
the row-fold probe must show row 8 stays on row 8, not row 0.
Second check:
the distinct one-hot probe must pass.
Third check:
the two-hot probe must still pass.
Fourth check:
test_isolate.py must pass all four cases.
Only after that, and with your separate fcloud approval for another remote round, I would sync/rebuild on fcloud and rerun the live probes.

7. Success and failure criteria:
Success:
row-fold probe passes, distinct one-hot passes, two-hot still passes, isolate passes.
Failure:
rows still alias across the `0..7` and `8..15` halves, or the fix breaks the previously good single-row accumulation path.

If you approve this proposal, I’ll make the smallest A-fragment edit first and validate it locally before asking to run the next fcloud retest.I’m turning the diagnosis into a concrete, narrow fix proposal first, because this repo requires proposal-plus-approval before any kernel edit.

I’m checking the SM120 hardware note before I lock the proposal, so the plan stays aligned with the repo’s kernel-change rules and with the actual MMA contract on this GPU.

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md)

I checked the SM120 hardware note in SM120_RTX_PRO_HARDWARE.md. It is consistent with our path: warp-level `mma.sync` on SM120 is the right primitive, so this is a correctness fix inside the existing warp-level FP8 MMA path, not a redesign.

**Proposal**
1. Objective and expected gain:
Fix the row-fold bug in w4a8_fp8_qmma.cu, where rows `8..15` of each `m16` tile are being accumulated into rows `0..7`. Expected gain is correctness recovery, not speed: the narrow success criterion is that the row-fold probe passes and test_isolate.py stops failing on random activations. If correct, this should move end-to-end accuracy from effectively unusable toward the prior dense GPTQ+FP8-KV baseline instead of `0%`.

2. Falsifiable local hypothesis:
The A fragment packing at w4a8_fp8_qmma.cu through w4a8_fp8_qmma.cu does not match the actual SM120 `m16n8k32.row.col` lane contract. The lower-half rows are not being represented as independent `row+8` outputs. That matches the live probe result where an activation only on row 8 produces output on row 0 and leaves row 8 zero.

3. Exact files/functions to change:
Source:
w4a8_fp8_qmma.cu
Function:
`w4a8_fp8_qmma_kernel`
Edit slice:
the A-fragment load block around w4a8_fp8_qmma.cu
Likely unchanged in the first attempt:
the B-fragment load block around w4a8_fp8_qmma.cu
Validation helpers:
either add or formalize the row-fold and basis probes under demo_sala
Required docs for this feature if approved:
`CHANGE_W4A8_001_iteration_003.en.md` and `.zh.md`
matching chat log under chat

4. Rule-compliance check:
This stays within the allowed submission path.
No eval harness changes.
No prepare-env launch bypass.
No switch away from the current baseline config.
No load-time full INT4→FP8 conversion in HBM.
Only submission-side kernel behavior changes.

5. Risk to accuracy and stability:
Main risk is that my first A-fragment remap is still not the exact PTX contract, in which case the row-fold bug may change shape rather than disappear.
Low risk of speed regression from the first edit because I plan to keep the same kernel structure and only change fragment packing.
Low risk to unrelated paths because the change is isolated to this kernel.

6. Validation plan:
First focused check after the first edit:
the row-fold probe must show row 8 stays on row 8, not row 0.
Second check:
the distinct one-hot probe must pass.
Third check:
the two-hot probe must still pass.
Fourth check:
test_isolate.py must pass all four cases.
Only after that, and with your separate fcloud approval for another remote round, I would sync/rebuild on fcloud and rerun the live probes.

7. Success and failure criteria:
Success:
row-fold probe passes, distinct one-hot passes, two-hot still passes, isolate passes.
Failure:
rows still alias across the `0..7` and `8..15` halves, or the fix breaks the previously good single-row accumulation path.

If you approve this proposal, I’ll make the smallest A-fragment edit first and validate it locally before asking to run the next fcloud retest.
