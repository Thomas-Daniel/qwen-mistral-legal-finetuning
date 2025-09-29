#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, argparse, random
import os
import torch
import flash_attn 
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

def set_attention_backend(model, attn: str = "flash2"):
    """
    attn: 'flash2' (FlashAttention 2), 'sdpa' (PyTorch fused), or 'eager' (fallback).
    Sets both HF model config and PyTorch SDPA kernel preferences.
    """
    attn = (attn or "flash2").lower()

    try:
        if attn == "flash2":
            model.config.attn_implementation = "flash_attention_2"
        elif attn == "sdpa":
            model.config.attn_implementation = "sdpa"
        else:
            model.config.attn_implementation = "eager"
    except Exception:
        pass

    try:
        if attn == "flash2":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        elif attn == "sdpa":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        else:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return model

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_chat_jsonl(dir_path: str, dataset: str, split: str, subset_rows: int = 0):
    p = os.path.join(dir_path, dataset, "chat", f"{split}.jsonl")
    assert os.path.exists(p), f"Missing file: {p}"
    rows = list(read_jsonl(p))
    if subset_rows and subset_rows > 0:
        rows = rows[:subset_rows]
    return Dataset.from_list(rows)

def build_prompter(tokenizer):
    def to_text(example):
        msgs = example["messages"]
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        example["text"] = rendered
        return example
    return to_text

def make_response_template(tokenizer):
    tmpl = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}],
        tokenize=False, add_generation_prompt=False
    )
    return "assistant" if "assistant" in tmpl else None

def ensure_chat_template(tokenizer, model_name: str):
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl and "{% generation" in tpl:  
        return

    mn = (model_name or "").lower()

    if "mistral" in mn:
        tokenizer.chat_template = r"""
{{ bos_token }}
{%- set sys = None -%}
{%- for m in messages -%}
  {%- if m['role'] == 'system' -%}{%- set sys = m['content'] -%}{%- endif -%}
{%- endfor -%}
{%- for m in messages -%}
  {%- if m['role'] == 'user' -%}
[INST] {{ ('<<SYS>>\n' ~ sys ~ '\n<</SYS>>\n\n') if sys and loop.first else '' }}{{ m['content'] }} [/INST]
  {%- elif m['role'] == 'assistant' -%}
  {%- if loop.last -%}{% generation %}{{ m['content'] }}{% endgeneration %}{{ eos_token }}
  {%- else -%}{{ m['content'] }}{{ eos_token }}
  {%- endif -%}
  {%- endif -%}
{%- endfor -%}
""".strip()
        return

    if "qwen2" in mn or "qwen" in mn:
        tokenizer.chat_template = r"""
{{ bos_token }}
{%- for i in range(messages|length) -%}
  {%- set m = messages[i] -%}
  {%- set is_last = (i == messages|length - 1) -%}
  {%- if is_last and m['role'] == 'assistant' -%}
<|im_start|>assistant
{% generation %}{{ m['content'] }}{% endgeneration %}<|im_end|>
  {%- else -%}
<|im_start|>{{ m['role'] }}
{{ m['content'] }}<|im_end|>
  {%- endif -%}
{%- endfor -%}

""".strip()
        return

    tokenizer.chat_template = r"""
{{ bos_token }}
{%- for m in messages -%}
  {%- if m['role'] == 'user' -%}
[INST] {{ m['content'] }} [/INST]
  {%- elif m['role'] == 'assistant' -%}
  {%- if loop.last -%}{% generation %}{{ m['content'] }}{% endgeneration %}{{ eos_token }}
  {%- else -%}{{ m['content'] }}{{ eos_token }}
  {%- endif -%}
  {%- endif -%}
{%- endfor -%}
""".strip()



def make_qlora_config(r=16, alpha=32, dropout=0.05):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

def make_dora_config(r=64, alpha=16, dropout=0.05):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "wi", "wo"
        ],
        use_dora=True
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True,
                    help="e.g. Qwen/Qwen2-7B-Instruct or mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root dir containing ledgar/ and eurlex/ subfolders")
    ap.add_argument("--dataset", type=str, required=True, choices=["eurlex", "ledgar"])
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--peft", type=str, default="qlora", choices=["qlora", "dora", "none"])

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed JSON")
    ap.add_argument("--subset_rows", type=int, default=0, help="Take first N rows (smoke test)")
    ap.add_argument("--max_steps", type=int, default=-1, help="Override steps (smoke test). -1 disables")
    ap.add_argument("--no_assistant_only_loss", action="store_true",
                    help="Disable assistant-only masking (train on full text).")
    ap.add_argument("--attn", type=str, default="flash2",choices=["flash2", "sdpa", "eager"],help="Attention kernel: flash2 (FlashAttention 2), sdpa (PyTorch fused), or eager")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    load_in_4bit = args.peft == "qlora"
    torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ensure_chat_template(tokenizer, args.model_name)

    quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    from peft.utils.other import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    model = set_attention_backend(model, args.attn)

    if args.peft == "qlora":
        model = get_peft_model(model, make_qlora_config())
    elif args.peft == "dora":
        model = get_peft_model(model, make_dora_config())
    model.print_trainable_parameters()

    
    train_ds = load_chat_jsonl(args.data_root, args.dataset, "train", subset_rows=args.subset_rows)
    train_ds = train_ds.map(build_prompter(tokenizer)).shuffle(seed=args.seed)
    print("Sample rendered train text:\n", train_ds[0]["text"])
    
    use_assistant_only_loss = not args.no_assistant_only_loss
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        do_eval =False,
        prediction_loss_only=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps if args.max_steps >= 0 else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.peft == "qlora" else "adamw_torch",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        group_by_length=True,
        deepspeed=args.deepspeed_config,

        max_length=args.max_seq_len,
        packing=False,
        assistant_only_loss=use_assistant_only_loss,
        save_strategy="steps",
        save_steps=50,  
        report_to="tensorboard",       
    )
    task_name = "eurlex" if args.dataset == "eurlex" else "ledgar"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=sft_cfg,
)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.peft in ("qlora", "dora"):
        model.save_pretrained(os.path.join(args.output_dir, "adapter"))

if __name__ == "__main__":
    main()
