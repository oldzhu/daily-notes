INFO  {'process': 'gptq', 'layer': 31, 'module': 'self_attn.o_proj', 'feat: in, out': '4096, 4096', 'dtype: size': 'bf16: 33.0MB', 'loss': '0.0012192388', 'samples': '69076', 'damp': '0.05000', 'time': '0.423', 'fwd_time': '0.154', '(v)ram': 'cuda 18.15G', 'dynamic': None}
                                                                                                                                                                                       INFO  {'process': 'gptq', 'layer': 31, 'module': 'mlp.gate_proj', 'feat: in, out': '4096, 16384', 'dtype: size': 'bf16: 132.0MB', 'loss': '0.0028136535', 'samples': '69076', 'damp': '0.05000', 'time': '1.269', 'fwd_time': '0.378', '(v)ram': 'cuda 34.5G', 'dynamic': None}
                                                                                                                                                                                       INFO  {'process': 'gptq', 'layer': 31, 'module': 'mlp.up_proj', 'feat: in, out': '4096, 16384', 'dtype: size': 'bf16: 132.0MB', 'loss': '0.0031371994', 'samples': '69076', 'damp': '0.05000', 'time': '1.276', 'fwd_time': '0.378', '(v)ram': 'cuda 34.5G', 'dynamic': None}
                                                                                                                                                                                       INFO  {'process': 'gptq', 'layer': 31, 'module': 'mlp.down_proj', 'feat: in, out': '16384, 4096', 'dtype: size': 'bf16: 132.1MB', 'loss': '0.0306088428', 'samples': '69076', 'damp': '0.05000', 'time': '1.924', 'fwd_time': '0.963', '(v)ram': 'cuda 50.42G', 'dynamic': None}
                                                                                                                                                                                       INFO  tp-pre-pad summary:                                  
[]                                                                                                                                                          
Processor finalization 2/2 tp-pre-pad █████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 0:00:00 / 0:00:00 [2/2] 100.0%                                                                                                                                                                                       INFO  | Process quant      | 448   | 1.925  | 0.759 | 340.096 | 29.7%  | model.layers.31.mlp.down_proj                     |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Submodule finalize | 448   | 1.584  | 0.732 | 327.848 | 28.7%  | model.layers.31.mlp.gate_proj                     |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Finalize pack      | 224   | 1.156  | 0.566 | 126.819 | 11.1%  | model.layers.31.mlp.up_proj [module.pack_block]   |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Finalize create    | 224   | 0.368  | 0.460 | 102.929 | 9.0%   | model.layers.31.mlp.up_proj                       |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Turtle reload      | 33    | 2.227  | 2.254 | 74.375  | 6.5%   | auto:MiniCPMSALADecoderLayer                      |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Pre-quant forward  | 128   | 0.963  | 0.513 | 65.630  | 5.7%   | model.layers.31:subset4/4                         |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Forward hook       | 7168  | 0.001  | 0.006 | 45.350  | 4.0%   | model.layers.31.mlp.down_proj                     |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Finalize offload   | 224   | 0.043  | 0.195 | 43.726  | 3.8%   | model.layers.31.mlp.gate_proj                     |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Post-quant replay  | 31    | 0.316  | 0.374 | 11.585  | 1.0%   | model.layers.30:subset4/4                         |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Capture inputs     | 1     | 5.647  | 5.647 | 5.647   | 0.5%   | cache_inputs:MiniCPMSALADecoderLayer              |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Process finalize   | 2     | 0.000  | 0.000 | 0.001   | 0.0%   | tp-pre-pad                                        |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  Saved Quantize Config: 
{
  "bits": 4,
  "dynamic": {
    "-:.*self_attn\\.o_gate.*": {},
    "-:.*self_attn\\.z_proj.*": {}
  },
  "group_size": 128,
  "desc_act": false,
  "lm_head": false,
  "quant_method": "gptq",
  "checkpoint_format": "gptq",
  "pack_dtype": "int32",
  "meta": {
    "quantizer": [
      "gptqmodel:5.7.0"
    ],
    "uri": "https://github.com/modelcloud/gptqmodel",
    "damp_percent": 0.05,
    "damp_auto_increment": 0.01,
    "static_groups": false,
    "true_sequential": true,
    "mse": 0.0,
    "gptaq": null,
    "act_group_aware": true,
    "failsafe": {
      "strategy": "rtn",
      "threshold": "0.5%",
      "smooth": {
        "type": "mad",
        "group_size_threshold": 128,
        "k": 2.75
      }
    },
    "offload_to_disk": true,
    "offload_to_disk_path": "./gptqmodel_offload/hqdpgrum-rkaakpxx/",
    "pack_impl": "cpu",
    "mock_quantization": false,
    "gc_mode": "interval",
    "wait_for_submodule_finalizers": false,
    "auto_forward_data_parallel": true,
    "hessian": {
      "chunk_size": null,
      "chunk_bytes": null,
      "staging_dtype": "float32"
    },
    "vram_strategy": "exclusive"
  },
  "sym": true,
  "format": "gptq"
}
Files in directory:
quant_log.csv
modeling_minicpm_sala.py
configuration_minicpm_sala.py
config.json
generation_config.json
quantize_config.json
model.safetensors.index.json
chat_template.jinja
tokenizer_config.json
special_tokens_map.json
added_tokens.json
tokenizer.model
tokenizer.json
Content of saved `generation_config.json`:
{
    "_from_model_config": true,
    "bos_token_id": 1,
    "do_sample": true,
    "eos_token_id": [
        2,
        73440
    ],
    "pad_token_id": 2,
    "transformers_version": "4.57.1"
}
Content of saved `config.json`:
{
    "architectures": [
        "MiniCPMSALAForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_use_output_gate": true,
    "attn_use_rope": false,
    "auto_map": {
        "AutoConfig": "configuration_minicpm_sala.MiniCPMSALAConfig",
        "AutoModel": "modeling_minicpm_sala.MiniCPMSALAModel",
        "AutoModelForCausalLM": "modeling_minicpm_sala.MiniCPMSALAForCausalLM",
        "AutoModelForSeq2SeqLM": "modeling_minicpm_sala.MiniCPMSALAForCausalLM",
        "AutoModelForSequenceClassification": "modeling_minicpm_sala.MiniCPMSALAForSequenceClassification"
    },
    "bos_token_id": 1,
    "dim_model_base": 256,
    "dtype": "bfloat16",
    "eos_token_id": [
        2,
        73440
    ],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.1,
    "intermediate_size": 16384,
    "lightning_head_dim": 128,
    "lightning_nh": 32,
    "lightning_nkv": 32,
    "lightning_scale": "1/sqrt(d)",
    "lightning_use_rope": true,
    "max_position_embeddings": 524288,
    "mixer_types": [
        "minicpm4",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "minicpm4",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "minicpm4",
        "minicpm4",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "minicpm4",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "lightning-attn",
        "minicpm4",
        "minicpm4",
        "minicpm4"
    ],
    "model_type": "minicpm_sala",
    "mup_denominator": 32,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 2,
    "pad_token_id": 2,
    "pretraining_tp": 1,
    "qk_norm": true,
    "quantization_config": {
        "bits": 4,
        "checkpoint_format": "gptq",
        "desc_act": false,
        "dynamic": {
            "-:.*self_attn\\.o_gate.*": {},
            "-:.*self_attn\\.z_proj.*": {}
        },
        "format": "gptq",
        "group_size": 128,
        "lm_head": false,
        "meta": {
            "act_group_aware": true,
            "auto_forward_data_parallel": true,
            "damp_auto_increment": 0.01,
            "damp_percent": 0.05,
            "failsafe": {
                "smooth": {
                    "group_size_threshold": 128,
                    "k": 2.75,
                    "type": "mad"
                },
                "strategy": "rtn",
                "threshold": "0.5%"
            },
            "gc_mode": "interval",
            "gptaq": null,
            "hessian": {
                "chunk_bytes": null,
                "chunk_size": null,
                "staging_dtype": "float32"
            },
            "mock_quantization": false,
            "mse": 0.0,
            "offload_to_disk": true,
            "offload_to_disk_path": "./gptqmodel_offload/hqdpgrum-rkaakpxx/",
            "pack_impl": "cpu",
            "quantizer": [
                "gptqmodel:5.7.0"
            ],
            "static_groups": false,
            "true_sequential": true,
            "uri": "https://github.com/modelcloud/gptqmodel",
            "vram_strategy": "exclusive",
            "wait_for_submodule_finalizers": false
        },
        "pack_dtype": "int32",
        "quant_method": "gptq",
        "sym": true
    },
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "rope_theta": 10000.0,
    "scale_depth": 1.4,
    "scale_emb": 12,
    "shift_labels": true,
    "sparse_config": {
        "block_size": 64,
        "dense_len": 8192,
        "init_blocks": 1,
        "kernel_size": 32,
        "kernel_stride": 16,
        "topk": 64,
        "use_nope": false,
        "window_size": 2048
    },
    "tie_word_embeddings": false,
    "transformers_version": "4.57.1",
    "use_cache": true,
    "use_output_gate": true,
    "use_output_norm": true,
    "vocab_size": 73448
}
INFO  Module: Sync model.embed_tokens <- from turtle (Embedding)                                                                                                                      
INFO  Module: Sync model.norm <- from turtle (MiniCPMRMSNorm)                                                                                                                         
INFO  Module: Sync lm_head <- from turtle (Linear)                                                                                                                                    
INFO  Module: Total synced modules: 3                                                                                                                                                 
INFO  Pre-Quantized model size: 18076.38MB, 17.65GB                                                                                                                                   
INFO  Quantized model size: 6308.76MB, 6.16GB                                                                                                                                         
INFO  Size difference: 11767.62MB, 11.49GB - 65.10%                                                                                                                                   
INFO  | Process quant      | 448   | 1.925  | 0.759 | 340.096 | 29.7%  | model.layers.31.mlp.down_proj                     |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Submodule finalize | 448   | 1.584  | 0.732 | 327.848 | 28.6%  | model.layers.31.mlp.gate_proj                     |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Finalize pack      | 224   | 1.156  | 0.566 | 126.819 | 11.1%  | model.layers.31.mlp.up_proj [module.pack_block]   |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Finalize create    | 224   | 0.368  | 0.460 | 102.929 | 9.0%   | model.layers.31.mlp.up_proj                       |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Turtle reload      | 33    | 2.227  | 2.254 | 74.375  | 6.5%   | auto:MiniCPMSALADecoderLayer                      |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Pre-quant forward  | 128   | 0.963  | 0.513 | 65.630  | 5.7%   | model.layers.31:subset4/4                         |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Forward hook       | 7168  | 0.001  | 0.006 | 45.350  | 4.0%   | model.layers.31.mlp.down_proj                     |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Finalize offload   | 224   | 0.043  | 0.195 | 43.726  | 3.8%   | model.layers.31.mlp.gate_proj                     |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Post-quant replay  | 31    | 0.316  | 0.374 | 11.585  | 1.0%   | model.layers.30:subset4/4                         |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Capture inputs     | 1     | 5.647  | 5.647 | 5.647   | 0.5%   | cache_inputs:MiniCPMSALADecoderLayer              |                                                          
INFO  +--------------------+-------+--------+-------+---------+--------+---------------------------------------------------+                                                          
INFO  | Model save         | 1     | 2.765  | 2.765 | 2.765   | 0.2%   | /root/models/openbmb/MiniCPM-SALA-quant-outputs/quant_smoke_32 |                                             
INFO  +--------------------+-------+--------+-------+---------+--------+----------------------------------------------------------------+                                             
INFO  | Process finalize   | 2     | 0.000  | 0.000 | 0.001   | 0.0%   | tp-pre-pad                                                     |                                             
INFO  +--------------------+-------+--------+-------+---------+--------+----------------------------------------------------------------+                                             
[preprocess] mode=gptq done - quantized model saved to /root/models/openbmb/MiniCPM-SALA-quant-outputs/quant_smoke_32
===== done =====
Smoke output: /root/models/openbmb/MiniCPM-SALA-quant-outputs//quant_smoke_32
Smoke log:    /root/models/openbmb/MiniCPM-SALA-quant-outputs//smoke_32.log
