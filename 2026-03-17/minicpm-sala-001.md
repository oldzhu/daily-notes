Here is the exact Marlin config meaning first, then the blocker I found.

In gptq_marlin.cu:141, the current candidate tables are:

small_batch_thread_configs
{128, 128, 256}
{64, 128, 128}
{128, 64, 128}

large_batch_thread_configs
{64, 256, 256}
{64, 128, 128}
{128, 64, 128}

Each tuple is:

thread_k
This is the K-dimension tile size per threadblock in the quantized weight GEMM.
thread_n
This is the N-dimension tile size per threadblock, meaning how many output columns or output channels the threadblock covers.
num_threads
This is the CUDA threadblock size. 256 means 8 warps, 128 means 4 warps.
Then inside gptq_marlin.cu:472, determine_exec_config picks the first valid config from one of those tables. It does not benchmark, and it does not branch on SM120. So right now the kernel policy is generic.

The other important parameters are:

thread_m_blocks
This is how many 16x16 blocks of M are handled per threadblock. It is derived from runtime batch shape in marlin_mm, not hardcoded in the table.
thread_n_blocks = thread_n / 16
This is the kernel template N blocking factor.
thread_k_blocks = thread_k / 16
This is the kernel template K blocking factor.
m_block_size_8
A special path for very small M.
group_blocks
This is roughly group_size / 16 when act-order is not active, and it affects scale-group handling inside the kernel.
Why this matters for your 4-bit model:
the 4-bit GPTQ weights are still being used,
but Marlin is the fused runtime kernel that reads packed 4-bit weights, applies scales and layout transforms, and does the GEMM.
So changing these tables absolutely changes performance for the 4-bit model.

Now the blocker: your current submission package does not actually vendor the Marlin C++ source. The demo package installs editable Python from pyproject.toml:61, but that file still depends on pip-installed sgl-kernel==0.3.20. The demo package only contains sglang/python, not sgl-kernel sources. So if I patch gptq_marlin.cu, that change will affect your local repo, but it will not be included in the submission tarball under the current packaging model.

So CHANGE_0043 as “SM120-specific Marlin kernel tuning” is approved in principle, but I cannot safely implement it as a submission-ready feature without also changing how sgl-kernel is supplied. That turns it into a different feature than the one we just approved.

The two viable paths are:

Submission-ready path
Vendor or otherwise provide a patched sgl-kernel in the demo package, and update prepare_env.sh to install that patched kernel instead of plain sgl-kernel==0.3.20.
This is the correct competition path, but it is a larger packaging change.
Local-research path
Patch the repo-root Marlin source only, so you can build and benchmark it locally or on fcloud for research.
This is useful to learn the right SM120 config, but it is not directly shippable yet.
My recommendation is:
do not patch the root Marlin kernel blindly yet.
First make CHANGE_0043 a packaging feature:
“submission-ready patched sgl-kernel installation for SM120 Marlin tuning”.
Then the next iteration can be the actual SM120 table change.

If you want, I can immediately prepare that revised CHANGE_0043 proposal and implementation plan:

package a patched sgl-kernel path into the demo submission
update prepare_env.sh to install it
then add the first SM120 Marlin config-table tweak on top of that
That would be the correct, shippable route.

