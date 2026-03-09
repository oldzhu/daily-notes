
python3 - <<'PY'
from gptqmodel import GPTQModel, QuantizeConfig
import os

model_dir = "/root/models/openbmb/MiniCPM-SALA"

qcfg = QuantizeConfig(
    bits=4,
    group_size=128,
    dynamic={r"-:.*o_gate.*": {}},
)

model = GPTQModel.load(
    model_dir,
    qcfg,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

print("module_tree =", getattr(model, "module_tree", None))
print("simple_layer_modules =", model.simple_layer_modules(model.model.config, model.quantize_config))
print("full_layer_modules =", model.full_layer_modules(model.model.config))
PY
===
...
}
               
INFO  Kernel: loaded -> `[]`                                                                                                                                                          
module_tree = ['model', 'layers', '#', {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'o_gate'), 'mlp': ('gate_proj', 'up_proj', 'down_proj')}]
simple_layer_modules = [['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'], ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']]
full_layer_modules = [['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.o_gate'], ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']]
