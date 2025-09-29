#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, re, argparse,os
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

_CLAUSE_RE = re.compile(
    r"Clause:\s*(?:\{)?(?P<clause>.*?)(?:\})?\s*(?:\n+|$)Answer with a single label",
    re.IGNORECASE | re.DOTALL,
)

def extract_clause(input_text: str) -> str:
    m = _CLAUSE_RE.search(input_text)
    if m:
        clause = m.group("clause").strip()
        if clause.startswith("{") and clause.endswith("}"):
            clause = clause[1:-1].strip()
        return clause
    if "Clause:" in input_text:
        after = input_text.split("Clause:", 1)[1]
        stop = after.split("Answer with a single label", 1)[0]
        return stop.strip().strip("{}").strip()
    return input_text.strip()

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def normalize_pred(pred: str) -> str:
    pred = pred.strip()
    for sep in ["\n", ","]:
        if sep in pred:
            pred = pred.split(sep, 1)[0].strip()
    pred = pred.strip().strip("`").strip().strip("{}").strip()
    return pred

def nearest_label(pred: str, labels: List[str]) -> str:
    if pred in labels:
        return pred
    lower_map = {l.lower(): l for l in labels}
    if pred.lower() in lower_map:
        return lower_map[pred.lower()]
    cands = []
    for l in labels:
        lp = l.lower()
        p = pred.lower()
        score = 0.0
        if p == lp: score = 100.0
        if p in lp: score += 10 + len(p) / (len(lp)+1)
        if lp in p: score += 5 + len(lp) / (len(p)+1)
        s1, s2 = set(p.split()), set(lp.split())
        if s1 and s2:
            jacc = len(s1 & s2) / len(s1 | s2)
            score += 2.0 * jacc
        common_prefix = os.path.commonprefix([p, lp])
        score += len(common_prefix) / (max(len(lp),1))
        cands.append((score, l))
    cands.sort(reverse=True)
    return cands[0][1]

INSTR = ("Task: Classify the following contract clause into exactly one LEDGAR category. "
         "Allowed labels are listed above in the instruction. "
         "Respond with label names only, comma-separated for multiple labels. Do not add extra text.\n\n"
         "Clause:\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to your fine-tuned SFT checkpoint dir")
    ap.add_argument("--eval_jsonl", required=True, help="Alpaca-style eval jsonl")
    ap.add_argument("--labels_json", required=True, help="labels.json with {'labels': [...]} ")
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base_model", required=False, default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    label2id = {l:i for i,l in enumerate(labels)}

    records = list(read_jsonl(args.eval_jsonl))[:args.limit]
    clauses, gold = [], []
    for r in records:
        clauses.append(extract_clause(r.get("input","")))
        gold.append(r.get("output","").strip())


    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    preds = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(clauses), bs), desc="Evaluating"):
            batch = clauses[i:i+bs]
            prompts = [INSTR + c + "\n\nAnswer with a single label from the list above." for c in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_seq_len).to(args.device)

            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            outs = gen[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(outs, skip_special_tokens=True)
            for t in texts:
                n = normalize_pred(t)
                preds.append(nearest_label(n, labels))

    y_true = [label2id.get(x, -1) for x in gold]
    y_pred = [label2id.get(x, -1) for x in preds]

    keep = [k for k,(a,b) in enumerate(zip(y_true,y_pred)) if a!=-1 and b!=-1]
    y_true = [y_true[k] for k in keep]
    y_pred = [y_pred[k] for k in keep]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n# Results")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro-F1:   {macro_f1:.4f}")
    print(f"Samples:    {len(y_true)}")

if __name__ == "__main__":
    main()
