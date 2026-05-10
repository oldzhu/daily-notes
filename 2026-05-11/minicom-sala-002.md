Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "echo '=== tokenizer_config.json ==='; diff -u /root/models/openbmb/MiniCPM-SALA/tokenizer_config.json /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/tokenizer_config.json && echo IDENTICAL; echo; echo '=== generation_config.json ==='; diff -u /root/models/openbmb/MiniCPM-SALA/generation_config.json /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/generation_config.json && echo IDENTICAL; echo; echo '=== chat_template.jinja ==='; diff -u /root/models/openbmb/MiniCPM-SALA/chat_template.jinja /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/chat_template.jinja 2>&1 && echo IDENTICAL; echo; echo '=== file lists ==='; ls /root/models/openbmb/MiniCPM-SALA/ | sort; echo ---; ls /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/ | sort" 2>&1 | tail -120

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01E2adbMVLzUh3nGPmaSPcKm__vscode-1777415370024/content.txt)

**Confirmed mismatch.** Key findings:

1. **`tokenizer_config.json`**: GPTQ dir REMOVED the embedded `"chat_template"` key entirely. BF16 has it embedded; GPTQ has a separate `chat_template.jinja` file instead.
2. **`tokenizer_class`**: `LlamaTokenizer` (BF16) vs `LlamaTokenizerFast` (GPTQ) — could affect tokenization edge cases.
3. **GPTQ has a separate `chat_template.jinja`**, BF16 does not.

So when Round 13e ran (server = BF16, eval --model_path = GPTQ): HF transformers prefers `chat_template.jinja` if present, so eval used the GPTQ-side template; server used the BF16-embedded template. **Definite mismatch.** Let me check whether the two chat templates are functionally identical:   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "python3 -c "
import json, hashlib
bf16 = json.load(open('/root/models/openbmb/MiniCPM-SALA/tokenizer_config.json'))
embedded = bf16.get('chat_template', '')
gptq = open('/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/chat_template.jinja').read()
print('BF16 embedded len:', len(embedded), 'sha256:', hashlib.sha256(embedded.encode()).hexdigest()[:16])
print('GPTQ jinja file len:', len(gptq), 'sha256:', hashlib.sha256(gptq.encode()).hexdigest()[:16])
print('Equal:', embedded == gptq)
print('Equal stripped:', embedded.strip() == gptq.strip())
import difflib
diff = list(difflib.unified_diff(embedded.splitlines(), gptq.splitlines(), lineterm='', n=1))[:60]
print('--- first diff lines (BF16-embedded vs GPTQ-jinja):')
for d in diff: print(d)
"" 2>&1 | tail -80

Chat templates are **byte-identical** (sha256 match). The "embedded vs separate file" difference doesn't affect content. Now check whether the actual tokenization (and stop-token derivation) differs:    

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec "python3 -c "
from transformers import AutoTokenizer, GenerationConfig
bf16_p = '/root/models/openbmb/MiniCPM-SALA'
gptq_p = '/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8'
for label, p in [('BF16', bf16_p), ('GPTQ', gptq_p)]:
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    gc = GenerationConfig.from_pretrained(p)
    eos = gc.eos_token_id
    eos_l = [eos] if isinstance(eos, int) else (eos or [])
    stops = [tok.decode(t) for t in eos_l]
    if tok.eos_token: stops.append(tok.eos_token)
    print(f'=== {label} ===')
    print('  class:', type(tok).__name__)
    print('  eos_token_id:', eos)
    print('  eos_token:', repr(tok.eos_token))
    print('  bos_token:', repr(tok.bos_token))
    print('  stop_words:', sorted(set(s for s in stops if s)))
    msgs = [{'role':'user','content':'Hello, world! Test prompt.'}]
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok.encode(rendered)
    print('  rendered len:', len(rendered), 'ids len:', len(ids))
    print('  rendered[:200]:', repr(rendered[:200]))
    print('  ids[:20]:', ids[:20])
"" 2>&1 | tail -40

**End-to-end tokenization is identical.** Both dirs:
- Load as `LlamaTokenizerFast` (HF auto-upgrades the BF16 declaration)
- Same `eos_token_id=[2, 73440]`, same `eos_token='<|im_end|>'`, same `bos='<s>'`
- Same derived `stop_words=['</s>', '<|im_end|>']`
- Same rendered prompt (byte-identical 76 chars)
- Same token IDs

**Conclusion: the model_path mismatch did NOT cause the Round 13e timeouts.** The cosmetic diffs in `tokenizer_config.json` (`LlamaTokenizer`→`LlamaTokenizerFast` declaration, embedded vs separate `chat_template.jinja`, `_commit_hash: null`, `do_sample: true`) are all behaviorally inert — eval pops `do_sample`, HF picks the Fast variant for both, and the chat template content is byte-identical (sha256 match).

The Round 13e timeouts are therefore **genuinely sparse-attn long-context slowness**, not a tokenizer artifact.

Let me shut down fcloud and pause for your decision:     

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py shutdown 2>&1 | tail -5

fcloud shut down. Now updating CHANGE_0134 with the diagnosis result:     

Replacing 11 lines with 81 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0134_eval_model_path_quant_mode.en.md)


```

```


Replacing 10 lines with 74 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0134_eval_model_path_quant_mode.zh.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0134_eval_model_path_quant_mode.{en,zh}.md && git commit -m "docs(round-13e): CHANGE_0134 — diagnosis result, tokenizer mismatch ruled out

On-fcloud verification (2026-04-29) confirmed the two model dirs differ
in tokenizer_config.json/generation_config.json on the surface, but
end-to-end behavior (LlamaTokenizerFast for both, same chat template
sha256, same eos/stop_words, same prompt rendering, same token IDs) is
identical. Therefore the model_path mismatch did NOT cause the Round
13e Test 1 timeouts. Sparse-attn slowness at long context is the real
cause. CHANGE_0134 fix retained as good hygiene against future drift." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Committed and pushed `b8d8f63db`. fcloud is shut down.

## Diagnosis verdict

The model_path mismatch is **NOT** what caused the Round 13e timeouts. Despite surface differences in `tokenizer_config.json` (`LlamaTokenizer`→`LlamaTokenizerFast` declaration, embedded vs separate `chat_template.jinja`, `_commit_hash`, `do_sample: true`), the actual behavior is byte-identical: same `LlamaTokenizerFast` instance, same chat template content (sha256 match), same eos/stop_words, same rendered prompt (76 chars), same token IDs (17). The `do_sample: true` is popped by eval anyway.

Therefore the Round 13e timeouts at concurrency=32 long-context are **genuinely sparse-attention slowness** at bs=8 × max_seq=128K, not a tokenizer artifact.

CHANGE_0134 stays in as good hygiene (correct fix; protects against future drift), but it does not unblock anything. CHANGE_0133 (over-fill) also stays in (correct, removes 1-2 GB/step memory traffic), but is necessary-not-sufficient.

## Decision pending

Three options before next fcloud session:

- **A. Close sparse line** — continue dense optimization, current Test 12 baseline (GPTQ + FP8 KV + dense) remains best. Lowest cost.
- **B. Profile sparse decode** — cheap dive (single-request profile, no concurrency=32 burn) to identify the actual hot kernel (top-k scoring vs sparse FlashAttention vs metadata builders). Informs whether B/C are worth pursuing.
- **C. Try lower-concurrency sparse** — `--max-running-requests 4` instead of 8 to test bs scaling; rough cost ~30 min on fcloud.

My recommendation: **B → then decide**. A profile run is cheap and gives us actual data instead of guessing whether sparse can ever be made fast enough. Want me to draft a CHANGE_0135 profiling proposal?  



