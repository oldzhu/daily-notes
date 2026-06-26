Yes, for the current fused W4A8 REAL path, that is correct for M.

More precisely:

- In gptq.py, before calling the fused op, the code computes:
  - M_orig = x.size(0)
  - M_pad = ceil(M_orig / 128) * 128
- If M_orig is smaller than 128, it pads the activation rows up to 128.
- Then it calls the fused kernel with that padded activation tensor.
- After the kernel returns, it slices the output back to the original M_orig rows.

So for decode, where M is often 1, the current behavior is effectively:
- input rows: 1
- padded rows sent to fused kernel: 128
- output rows kept afterward: only the first 1

One important correction though: it is not padding everything to a literal 128 x 128 matrix in Python.

What is actually happening is:

- M is padded in the wrapper
- N is not padded in the wrapper
- K is not padded in the wrapper

The fused kernel in w4a8_fp8_qmma.cu is internally tiled as 128 x 128 x 64:
- tile M = 128
- tile N = 128
- tile K = 64

But only M is explicitly padded at the Python dispatch layer. For N and K:

- N is handled by grid ceil-division plus bounds checks inside the kernel
- K is handled by the loop over K tiles plus bounds checks

In this specific W4A8 REAL path, eligible layers are also filtered earlier so that the layer dimensions are multiples of 128 in gptq.py, so in practice the fused path usually sees N and K already aligned.

So the clean summary is:

- Yes, if M is smaller than 128, the current wrapper pads M to 128 before calling fused.
- No, it is not generally creating a fake full 128 x 128 matrix at the wrapper level.
- The problematic part for decode is exactly that M=1 becomes M=128 work.
