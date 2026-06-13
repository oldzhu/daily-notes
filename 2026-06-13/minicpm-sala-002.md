The accuracy test was making progress (up to 113/150). Let me check if it completed:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls -lt /root/data/outputs/ 2>&1 | head -5; echo "---"; ls /root/data/outputs/20260519_024330/ 2>&1; echo "---"; cat /root/data/outputs/20260519_024330/predictions.jsonl 2>/dev/null | wc -l' 2>&1

Server crashed again under load (at ~75% through accuracy). The W4A8 path works for individual requests but crashes under concurrency. Let me provide final summary:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1



## Session Summary

### ✅ Achieved
- **W4A8 REAL boots** with coherent output: *"France and is located in the north-"*
- **Root cause FOUND**: Marlin repack empties `qzeros` to `[0]` — fixed by moving dequant to setup time (before repack)
- **Used proven functions**: `gptq_int4_dequantize` + `fp8_blockwise_quantize` from `utils_w4a8_fp8` 

### ❌ Remaining
- **Server OOM under concurrency**: accuracy test reached 75% (113/150) then server crashed
- **Speed tests**: not yet run (will do S1→S8→Smax sequentially as you requested)
- **sgl-kernel rebuild**: still using old wheel (no CUDA dequant) — Python fallback works but slower

### Next Session Plan
1. Debug the concurrency stability (likely memory leak from FP8 weight cache)
2. Run accuracy to completion
3. Run S1/S8/Smax speed tests **one by one**
4. Rebuild sgl-kernel with CUDA dequant for optimal speed
