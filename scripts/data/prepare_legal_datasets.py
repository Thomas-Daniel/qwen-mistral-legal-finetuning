#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import os
import re
from hashlib import md5
from typing import Dict, List

from datasets import load_dataset, DatasetDict
from tqdm import tqdm



def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def truncate_chars(s: str, max_chars: int) -> str:
    if max_chars is None or max_chars <= 0:
        return s
    return s[:max_chars].rstrip()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def deduplicate_by_text(rows: List[Dict], text_key: str = "text") -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        h = md5(r[text_key].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


SYSTEM_PROMPT = (
    "You are a precise legal assistant. "
    "Follow instructions carefully. Respond with label names only, "
    "comma-separated for multiple labels. Do not add extra text."
)

def make_ledgar_user_prompt(label_list: List[str]) -> str:
    label_str = ", ".join(label_list)
    return (
        "Task: Classify the following contract clause into exactly one LEDGAR category.\n"
        f"Allowed labels:\n{label_str}\n\n"
        "Clause:\n{{text}}\n\n"
        "Answer with a single label from the list above."
    )

def make_eurlex_user_prompt(label_list: List[str]) -> str:
    label_str = ", ".join(label_list)
    return (
        "Task: Assign all applicable EUR-LEX legal topics to the following document (multi-label).\n"
        f"Allowed labels:\n{label_str}\n\n"
        "Document:\n{{text}}\n\n"
        "Answer with one or more labels, comma-separated, using only names from the list above."
    )

def convert_ledgar(split_ds, label_names: List[str], max_chars: int):
    chat_rows, cls_rows = [], []
    user_template = make_ledgar_user_prompt(label_names)
    for ex in tqdm(split_ds, desc="LEDGAR examples"):
        text = truncate_chars(clean_text(ex["text"]), max_chars)
        gold_label = label_names[ex["label"]]
        chat_rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_template.replace("{text}", text)},
                {"role": "assistant", "content": gold_label}
            ],
            "meta": {
                "dataset": "ledgar",
                "task": "single_label",
            }
        })
        cls_rows.append({
            "text": text,
            "labels": [gold_label],
        })

    chat_rows = deduplicate_by_text(
        [{"text": r["messages"][1]["content"], **r} for r in chat_rows], text_key="text"
    )
    for r in chat_rows:
        r.pop("text", None)
    cls_rows = deduplicate_by_text(cls_rows, text_key="text")
    return chat_rows, cls_rows


def convert_eurlex(split_ds, label_names: List[str], max_chars: int):
    chat_rows, cls_rows = [], []
    user_template = make_eurlex_user_prompt(label_names)
    for ex in tqdm(split_ds, desc="EUR-LEX examples"):
        text = truncate_chars(clean_text(ex["text"]), max_chars)
        gold_label_ids: List[int] = ex["labels"]
        gold = [label_names[i] for i in gold_label_ids]
        chat_rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_template.replace("{text}", text)},
                {"role": "assistant", "content": ", ".join(gold) if gold else "None"}
            ],
            "meta": {
                "dataset": "eurlex",
                "task": "multi_label",
            }
        })
        cls_rows.append({
            "text": text,
            "labels": gold,  
        })
    chat_rows = deduplicate_by_text(
        [{"text": r["messages"][1]["content"], **r} for r in chat_rows], text_key="text"
    )
    for r in chat_rows:
        r.pop("text", None)

    cls_rows = deduplicate_by_text(cls_rows, text_key="text")
    return chat_rows, cls_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Output root directory.")
    ap.add_argument("--max_chars", type=int, default=6000, help="Truncate texts to this many characters (0 = no limit).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading LEDGAR (lex_glue/ledgar)...")
    ledgar: DatasetDict = load_dataset("lex_glue", "ledgar")
    ledgar_label_names = ledgar["train"].features["label"].names
    print(f"LEDGAR labels: {len(ledgar_label_names)} classes")

    for split in ["train", "validation", "test"]:
        chat_rows, cls_rows = convert_ledgar(ledgar[split], ledgar_label_names, args.max_chars)
        base = os.path.join(args.out_dir, "ledgar")
        chat_dir = os.path.join(base, "chat")
        cls_dir = os.path.join(base, "classification")
        ensure_dir(chat_dir)
        ensure_dir(cls_dir)
        to_jsonl(os.path.join(chat_dir, f"{split}.jsonl"), chat_rows)
        to_jsonl(os.path.join(cls_dir, f"{split}.jsonl"), cls_rows)

    ledgar_base = os.path.join(args.out_dir, "ledgar")
    with open(os.path.join(ledgar_base, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": ledgar_label_names}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ledgar_base, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# LEDGAR (prepared)\n\n"
            "- Source: `lex_glue/ledgar`\n"
            "- Task: single-label contract clause classification (100 classes)\n"
            "- Exports:\n"
            "  - `chat/*.jsonl` (system/user/assistant for SFT)\n"
            "  - `classification/*.jsonl` with `{text, labels}` (labels are names)\n"
        )

    print("Loading EUR-LEX (lex_glue/eurlex)...")
    eurlex: DatasetDict = load_dataset("lex_glue", "eurlex")
    eurlex_label_names = eurlex["train"].features["labels"].feature.names
    print(f"EUR-LEX labels: {len(eurlex_label_names)} classes")
    for split in ["train", "validation", "test"]:
        chat_rows, cls_rows = convert_eurlex(eurlex[split], eurlex_label_names, args.max_chars)
        base = os.path.join(args.out_dir, "eurlex")
        chat_dir = os.path.join(base, "chat")
        cls_dir = os.path.join(base, "classification")
        ensure_dir(chat_dir)
        ensure_dir(cls_dir)
        to_jsonl(os.path.join(chat_dir, f"{split}.jsonl"), chat_rows)
        to_jsonl(os.path.join(cls_dir, f"{split}.jsonl"), cls_rows)

    eurlex_base = os.path.join(args.out_dir, "eurlex")
    with open(os.path.join(eurlex_base, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": eurlex_label_names}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(eurlex_base, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# EUR-LEX (prepared)\n\n"
            "- Source: `lex_glue/eurlex`\n"
            "- Task: multi-label legal topic assignment (21 labels)\n"
            "- Exports:\n"
            "  - `chat/*.jsonl` (system/user/assistant for SFT)\n"
            "  - `classification/*.jsonl` with `{text, labels}` (labels are names)\n"
            "- Notes: Assistant target contains comma-separated label names. Empty label sets are serialized as `None` in the chat target.\n"
        )

    print("\nDone. Files written under:", args.out_dir)
    print("Next steps:")
    print("- Point your SFT trainer (Q-LoRA / DoRA) to the chat JSONL files.")
    print("- Use classification JSONL files to build a clean evaluation pipeline.")


if __name__ == "__main__":
    main()
