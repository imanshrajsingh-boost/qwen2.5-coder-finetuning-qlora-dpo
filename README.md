---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
tags:
  - qwen2
  - code
  - sft
  - dpo
  - fine-tuned
  - autoawq
  - awq
  - quantized
  - conversational
  - text-generation-inference
datasets:
  - iamtarun/python_code_instructions_18k_alpaca
  - Vezora/Code-Preference-Pairs
language:
  - en
pipeline_tag: text-generation
---

# Qwen2.5-Coder-1.5B-AWQ (Custom Fine-Tuned & DPO Aligned)

A 4-bit AWQ-quantized, custom fine-tuned version of **Qwen2.5-Coder-1.5B-Instruct**, optimized for Python code generation, bug fixing, and reasoning tasks. This model went through a two-stage alignment pipeline (SFT followed by DPO) to improve both instruction-following and code quality/preference alignment, then was quantized to 4-bit AWQ for faster, lighter-weight inference.

## Model Details

| | |
|---|---|
| **Base model** | [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) |
| **Model size** | ~1.5B parameters (2B incl. quantization overhead) |
| **Quantization** | 4-bit AWQ, group size 128 |
| **Primary use case** | Python code generation, debugging, and code reasoning |
| **License** | Apache 2.0 |
| **Language** | English (code + natural language instructions) |

## Training Pipeline

This model was produced by a production-style **QLoRA → SFT → DPO → AWQ** pipeline, designed to run end-to-end on a single free-tier Colab T4 GPU (16GB VRAM). Full training code is in [`production_code_llm_finetuning.ipynb`](./production_code_llm_finetuning.ipynb) in this repo.

**1. QLoRA setup**
The base model is loaded in 4-bit NF4 precision (`BitsAndBytesConfig`, double quantization enabled, bf16 compute dtype), then wrapped with a LoRA adapter:
- Rank `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj` (attention) **and** `gate_proj, up_proj, down_proj` (MLP) — targeting MLP layers as well as attention is a deliberate choice for better downstream quality.

**2. Stage 1 — Supervised Fine-Tuning (SFT)**
Trained on [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca), deduplicated by instruction (MD5 hash) before training. Data was reformatted into chat-style `messages` (user instruction [+ optional input] → assistant output) and trained with `SFTTrainer`:
- 1 epoch, effective batch size 32 (`per_device=4` × `grad_accum=8`)
- LR `2e-4`, cosine schedule, `paged_adamw_8bit` optimizer, bf16
- Sequence packing enabled, max length 1024 tokens, `assistant_only_loss=True` (loss computed only on the assistant's response)
- Best checkpoint restored via eval loss (`eval_steps=50`)

The SFT adapter is then merged into the base weights to create a standalone reference model for Stage 2.

> **Note on dataset size:** training on the *full* 18.6k-example SFT set takes ~13 hours on a free Colab T4. For this release, a shuffled subset was used to fit within Colab's free-tier session limits. The notebook includes a commented-out `shuffle().select(range(...))` line — the full pipeline is there and works end-to-end; uncomment it (and run on a faster GPU, or a paid Colab tier) to train on the complete dataset for a stronger model.

**3. Stage 2 — Direct Preference Optimization (DPO)**
A fresh LoRA adapter (same config as above) is attached to the merged SFT model and trained on [`Vezora/Code-Preference-Pairs`](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs), reformatted into `prompt` / `chosen` / `rejected` triples:
- 1 epoch (DPO is more prone to collapse with more epochs), batch size 8 (effective, via grad accumulation)
- LR `5e-6` (deliberately much lower than SFT — DPO is more sensitive), `beta=0.1`
- No explicit reference model passed — TRL disables the adapter to use the merged SFT weights as the implicit reference, saving memory

> Same note applies here: the full 54k-pair preference set takes ~10 hours to train on a T4, so this release used a shuffled subset. The dataset-selection line is commented out in the notebook for anyone who wants to scale up to the full set.

**4. Final merge**
The DPO adapter is merged back into the SFT-merged weights to produce a single standalone, deployable checkpoint (no PEFT/adapter overhead at inference time).

**5. Evaluation**
- **Qualitative checks:** greedy/low-temperature (0.2) generations on held-out prompts (e.g. memoized Fibonacci, palindrome check), verified for chat-template adherence and code correctness — outputs were clean and syntactically correct.
- **DPO training metrics:** by the final logged step, `rewards/accuracies` reached **1.000** (the model consistently scores the chosen response higher than the rejected one on eval data) with a growing `rewards/margins` (~0.12), indicating the preference signal was learned rather than just memorized.
- **Standardized benchmarks:** the notebook includes a [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) run on **HumanEval** and **MBPP** (pass@1, real code execution) to measure functional correctness against the base model. This benchmark run did not finish in the Colab session this release was trained in (it was still loading weights when the runtime ended) — the eval harness command and config are fully set up in the notebook for anyone who wants to complete this run and get final pass@1 numbers.

**6. Quantization (AWQ)**
QLoRA's 4-bit weights are training-time only and not inference-optimized. For deployment, the merged fp/bf16 model is separately quantized with **AWQ** (Activation-aware Weight Quantization, group size 128) — calibration-based quantization that protects salient weights, giving ~4x smaller size and higher inference throughput with negligible accuracy loss. AWQ is natively supported by vLLM, TGI, and SGLang. This step completed successfully, producing the deployment-ready 4-bit model published in this repo.

> **Note:** AutoAWQ (the library used for quantization here) has been officially deprecated by its maintainer as of this pipeline's last tested configuration (Torch 2.6.0, Transformers 4.51.3). Ongoing AWQ tooling development has moved to the [vLLM Project's `llm-compressor`](https://github.com/vllm-project/llm-compressor) — worth switching to for future runs.

> **Tip:** for best output quality, upgrade `torch` in the second cell before running the full pipeline.

## How to Use

### Using Transformers & AutoAWQ

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Anshrajsingh/qwen2.5-coder-1.5b-awq"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)

prompt = "Write a Python function that returns the nth Fibonacci number using memoization."
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False
).to(model.device)

outputs = model.generate(input_ids, max_new_tokens=300, temperature=0.2)
print(tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True))
```

### Using vLLM (OpenAI-compatible server)

```bash
pip install vllm
vllm serve "Anshrajsingh/qwen2.5-coder-1.5b-awq"
```

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "Anshrajsingh/qwen2.5-coder-1.5b-awq",
    "messages": [
      {"role": "user", "content": "Write a Python function to reverse a linked list."}
    ]
  }'
```

### Using SGLang

```bash
pip install sglang
python3 -m sglang.launch_server \
    --model-path "Anshrajsingh/qwen2.5-coder-1.5b-awq" \
    --host 0.0.0.0 \
    --port 30000
```

## Reproducing This Model

The full pipeline — dependency setup, QLoRA config, SFT, DPO, evaluation, and AWQ quantization — is available as a single Colab-ready notebook: [`production_code_llm_finetuning.ipynb`](./production_code_llm_finetuning.ipynb). It's designed to run start-to-finish on a free-tier T4 GPU for the 1.5B model (use an A100/H100 for 7B+ variants).

## Intended Use

- Python code generation from natural-language prompts
- Debugging and fixing existing code snippets
- Explaining or reasoning about code logic
- Lightweight, low-latency coding assistant for local/edge deployment (thanks to 4-bit AWQ quantization)

## Limitations

- Optimized primarily for **Python**; performance on other languages is not guaranteed and hasn't been specifically evaluated.
- As a 1.5B-parameter model, it will not match the reasoning depth of larger code models (e.g., 7B+ variants) on complex, multi-file, or highly abstract tasks.
- DPO alignment reflects the preferences encoded in the `Vezora/Code-Preference-Pairs` dataset, which may not match every user's stylistic preferences.
- Like any LLM, outputs should be reviewed before use in production — the model can still generate incorrect or insecure code.

## Citation

If you use this model, please consider citing the base model and datasets it builds on:

```bibtex
@misc{qwen2.5-coder-1.5b-awq,
  author = {Anshrajsingh},
  title = {Qwen2.5-Coder-1.5B-AWQ (Custom Fine-Tuned & DPO Aligned)},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Anshrajsingh/qwen2.5-coder-1.5b-awq}}
}
```

Base model: [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
