#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, argparse, random, os, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_alpaca_jsonl(dir_path: str, dataset: str, split: str, subset_rows: int = 0):
    p = os.path.join(dir_path, dataset, "chat", f"{split}_alpaca.jsonl")
    assert os.path.exists(p), f"Missing file: {p}"
    rows = list(read_jsonl(p))
    if subset_rows and subset_rows > 0:
        rows = rows[:subset_rows]
    return Dataset.from_list(rows)

def formatting_prompts_func(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )
    completion = output + "</s>"
    return {"prompt": prompt, "completion": completion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["eurlex","ledgar"])
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--lr", type=float, default=2e-5)          
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset_rows", type=int, default=0)

    ap.add_argument("--flash_attn", action="store_true", help="Use FlashAttention-2 if available.")
    ap.add_argument("--no_checkpointing", action="store_true", help="Disable gradient checkpointing for speed.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,   
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False  

    train_ds = load_alpaca_jsonl(args.data_root, args.dataset, "train", subset_rows=args.subset_rows)
    train_ds = train_ds.map(formatting_prompts_func).shuffle(seed=args.seed)
    train_ds = train_ds.shuffle(seed=args.seed).select(range(5000))
    print("Sample rendered train completion:\n", train_ds[0]["completion"])


    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,   
        fp16=False,
        gradient_checkpointing=True,      
        group_by_length=True,  
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        max_length=args.max_seq_len,     
        completion_only_loss=True,       
        packing=False,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=125,
        save_total_limit=8,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=sft_cfg,
        formatting_func=None,  
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
