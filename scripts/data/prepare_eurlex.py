#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import os
import re
from hashlib import md5
from typing import Dict, List, Optional

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

def make_eurlex_user_prompt(label_list: List[str]) -> str:
    label_str = ", ".join(label_list)
    return (
        "Task: Assign all applicable EUR-LEX legal topics to the following document (multi-label).\n"
        f"Allowed labels:\n{label_str}\n\n"
        "Document:\n{{text}}\n\n"
        "Answer with one or more labels, comma-separated, using only names from the list above."
    )

def load_descriptor_map(path: Optional[str], lang: str = "en") -> Optional[Dict[str, str]]:

    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    id2desc: Dict[str, str] = {}
    for k, v in obj.items():
        key = str(k)
        if isinstance(v, dict):
            if lang in v and isinstance(v[lang], str) and v[lang].strip():
                id2desc[key] = v[lang].strip()
            else:
                for cand_lang in [lang, "en"]:
                    if cand_lang in v and isinstance(v[cand_lang], str) and v[cand_lang].strip():
                        id2desc[key] = v[cand_lang].strip()
                        break
                if key not in id2desc:
                    for vv in v.values():
                        if isinstance(vv, str) and vv.strip():
                            id2desc[key] = vv.strip()
                            break
                    if key not in id2desc:
                        id2desc[key] = key 
        elif isinstance(v, str):
            id2desc[key] = v.strip() if v.strip() else key
        else:
            id2desc[key] = key
    return id2desc

def convert_eurlex(split_ds, label_names: List[str], max_chars: int, id2desc: Optional[Dict[str, str]]):
    chat_rows, cls_rows = [], []
    idx2id = {int(i): str(name) for i, name in enumerate(label_names)}
    if id2desc:
        allowed_names = [id2desc.get(idx2id[i], idx2id[i]) for i in range(len(label_names))]
        label_space_kind = "descriptors"
    else:
        allowed_names = [idx2id[i] for i in range(len(label_names))]
        label_space_kind = "eurovoc_ids"

    user_template = make_eurlex_user_prompt(allowed_names)

    for ex in tqdm(split_ds, desc="EUR-LEX examples"):
        text = truncate_chars(clean_text(ex["text"]), max_chars)
        gold_label_ids = ex.get("labels") or []
        eurovoc_ids = [idx2id[int(i)] for i in gold_label_ids]
        if id2desc:
            gold = [id2desc.get(eid, eid) for eid in eurovoc_ids]
        else:
            gold = eurovoc_ids
        chat_rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_template.replace("{text}", text)},
                {"role": "assistant", "content": ", ".join(gold) if gold else "None"}
            ],
            "meta": {"dataset": "eurlex", "task": "multi_label", "label_space": label_space_kind}
        })
        cls_rows.append({"text": text, "labels": gold})
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
    ap.add_argument("--descriptor_map", type=str, default=None,
                    help="Path to eurovoc_descriptors.json mapping EuroVoc IDs → descriptors. If omitted, keep IDs.")
    ap.add_argument("--lang", type=str, default="en",
                    help="Language key to extract from descriptor_map when it contains multilingual entries (default: en).")
    args = ap.parse_args()

    print("Loading EUR-LEX (lex_glue/eurlex)...")
    eurlex: DatasetDict = load_dataset("lex_glue", "eurlex")
    label_names = eurlex["train"].features["labels"].feature.names  
    print(f"- Label space size: {len(label_names)} EuroVoc concepts")

    id2desc = load_descriptor_map(args.descriptor_map, lang=args.lang) if args.descriptor_map else None
    if id2desc:
        print(f"- Loaded {len(id2desc)} EuroVoc ID→descriptor pairs (lang='{args.lang}'). Using descriptors.")
    else:
        print("- No descriptor map provided or readable. Using EuroVoc IDs as label names.")

    for split in ["train", "validation", "test"]:
        chat_rows, cls_rows = convert_eurlex(eurlex[split], label_names, args.max_chars, id2desc)

        base = os.path.join(args.out_dir, "eurlex")
        chat_dir = os.path.join(base, "chat"); ensure_dir(chat_dir)
        cls_dir = os.path.join(base, "classification"); ensure_dir(cls_dir)

        to_jsonl(os.path.join(chat_dir, f"{split}.jsonl"), chat_rows)
        to_jsonl(os.path.join(cls_dir, f"{split}.jsonl"), cls_rows)
        print(f"Wrote {len(chat_rows)} chat and {len(cls_rows)} classification rows for split='{split}'")

    if id2desc:
        used_labels = [id2desc.get(str(n), str(n)) for n in label_names]
        inv_type = "descriptors"
    else:
        used_labels = list(map(str, label_names))
        inv_type = "eurovoc_ids"

    base = os.path.join(args.out_dir, "eurlex"); ensure_dir(base)
    with open(os.path.join(base, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"type": inv_type, "labels": used_labels}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(base, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# EUR-LEX (prepared)\n\n"
            "- Source: `lex_glue/eurlex`\n"
            "- Task: multi-label topic classification (EuroVoc)\n"
            f"- Label space: {'English descriptors (via descriptor_map)' if id2desc else 'EuroVoc IDs (numeric strings)'}\n"
            "- Exports:\n"
            "  - `chat/*.jsonl` (system/user/assistant for SFT)\n"
            "  - `classification/*.jsonl` with `{text, labels}` (labels are names in the chosen space)\n"
            "- Notes: Assistant targets are comma-separated labels; empty label sets become `None` in chat.\n"
        )

    print("\nDone. Verify a couple lines:")
    print(f"- {os.path.join(args.out_dir, 'eurlex', 'classification', 'train.jsonl')}")
    print("  Should show descriptor strings if --descriptor_map is provided; otherwise EuroVoc IDs.")


if __name__ == "__main__":
    main()
