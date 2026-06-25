I’m checking the two kernel entry points directly so I can answer from the actual implementation rather than from inference.

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#941-941), lines 941 to 1005

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#59-59), lines 59 to 170

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#700-700), lines 700 to 910

I have the core reason already; I’m grabbing one more Marlin snippet so I can show it’s designed around arbitrary `prob_m`, not just infer that from the wrapper.

Searched for regex `prob_m|blocks|grid|small_batch|large_batch` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), 80 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#880-880), lines 880 to 1040

I confirmed the reason from the code: the fused path pads because this specific kernel was written as a fixed `128x128x64` tile kernel with an exact `M % 128 == 0` contract, while Marlin was built to handle arbitrary `size_m` and has small-batch/tail logic.



Because the current fused kernel is much more specialized than Marlin.

In the fused W4A8 path, the kernel in w4a8_fp8_qmma.cu hard-codes:

- `kTileM = 128`
- `kTileN = 128`
- `kTileK = 64`

and the entry point explicitly enforces:

- `TORCH_CHECK(M % kTileM == 0, "M must be multiple of ", kTileM);`

in w4a8_fp8_qmma.cu.  
Then the Python dispatch layer works around that by padding `x` up to a multiple of 128 before calling the fused op in gptq.py.

So the short answer is: fused pads because this implementation chose a fixed-tile contract for simplicity, not because FP8 MMA fundamentally requires padding.

Marlin is different. Its kernel path accepts arbitrary `size_m` directly in gptq_marlin.cu, and the main launcher logic in gptq_marlin.cu is built around `prob_m`, `div_ceil(prob_m, ...)`, split handling, and separate small-batch configs. You can see dedicated `small_batch_thread_configs` and `small_batch_thread_configs_sm120` in gptq_marlin.cu. In other words, Marlin was engineered to support tails and small `M`; the fused kernel was not.

There is one important nuance: the fused kernel body already has bounds checks on loads and stores, so in principle it could be changed to support non-multiple-of-128 `M` by launching `ceil_div(M, 128)` blocks and dropping the strict check. But the current entry point does not do that. Right now padding is an implementation choice and very likely part of why decode is slow, because decode often has `M=1`, and that gets inflated to `M=128`.

So if your question is “does fused have to pad?”, the answer is no.  
If your question is “why does this fused code pad while Marlin doesn’t?”, the answer is: this fused kernel is a first-cut fixed-tile kernel, while Marlin is a mature general-purpose kernel with tail handling and small-batch tuning.

The concrete implication is that your proposed next direction is valid: compare execution flow with fused enabled vs disabled, because the first obvious structural penalty is the `M=1 -> 128` padding in decode.
