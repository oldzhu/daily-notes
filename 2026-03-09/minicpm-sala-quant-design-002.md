In the currently proposed next step, it would remove o_gate from the GPTQ module tree globally, not only for the lightning layers.
Reason:

The failure comes from GPTQ building one shared per-layer module plan from layer0, then reusing that plan on later layers.
Because that shared plan contains self_attn.o_gate, any later layer without o_gate fails.
The simplest robust fix is to remove o_gate from the quantization target set entirely, so GPTQ never expects it on any layer.
What it does not do:

It does not delete o_gate from the model.
It only stops GPTQ from trying to quantize/calibrate that module.
Yes, it does affect calibration coverage, but in a limited and usually acceptable way.
Effect:

o_gate would no longer participate in GPTQ calibration statistics.
On layers where o_gate exists, that module would remain in original precision instead of being quantized.
All the other targeted modules would still be calibrated and quantized normally.
Tradeoff:

Upside: preprocessing stops failing on mixed sparse/lightning layers.
Downside: you lose quantization coverage on o_gate for the layers that do have it, so there may be a small memory/speed benefit left on the table.
My engineering judgment:

This is the lower-risk unblocker.
It is better to quantize 7 stable modules across all layers than to fail because of the 8th module only present in some layers.
If the model quality remains good and speed is acceptable, this is likely a practical submission path.
If you want the more precise version, the alternative is:

Detect per-layer structure.
Quantize o_gate only on sparse layers where it exists.
Exclude it on lightning layers.
That is more correct architecturally, but also more invasive and more likely to fight GPTQModel internals.

So the decision is:

Conservative unblocker: remove o_gate globally from GPTQ targeting.
Higher-complexity optimization: per-layer selective targeting.
If you approve, I recommend we take the conservative unblocker first, measure correctness/speed, and only revisit selective o_gate quantization if the measured gain is clearly worth the added complexity.


