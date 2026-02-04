I am  really interest and want to deep dive into the following area:
1. python glue _cutom_ops.py how it map to and called by the gpt inference operations(tokonizer, embeding, mha,mlp Layer Normalization, Positional Encoding or any other components or operations in gpt inferences)
2. how performance-critical kernels exposed in _custom_ops implemeneted and how it speed up the performance. comparing with the execution scenario without these customized kernels, how they improve the performance. 
3. The same questions to the cpu native kernels + careful memory layout + kernel fusion + graph capture. How they exposed to python and how they improve the things comaring with the scenario wuthout them.
be
pls review and reply, better to combine with some samples, execution flow graph and give vivid explaination and  guide so a user have deep and full understanding on that part in order to contribute on that part later .  pls also create the document for the guide and explanation.
