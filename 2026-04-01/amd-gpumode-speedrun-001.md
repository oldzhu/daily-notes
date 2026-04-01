Some comments on the compiler intrinsics/builtins and where to find docs/examples:

In most cases builtin translates into intrinsic. In most cases translation is 1:1. But not always. Many intrinsics directly or nearly directrly map to the ISA manual / shader programming guide insts
LLVM user guide https://llvm.org/docs/AMDGPUUsage.html
ISA https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
Builtins: https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td
Intrinsics descriptions https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
Code examples: https://github.com/carlushuang/gcnasm/tree/master
https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU
https://github.com/llvm/llvm-project/tree/main/clang/test/CodeGenOpenCL and https://github.com/llvm/llvm-project/tree/main/clang/test/SemaOpenCL
grep llvm repo
experiments
