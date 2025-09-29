import json
import os

data_root = "./data/processed/ledgar/chat"
splits = ["train.jsonl", "eval.jsonl", "test.jsonl","validation.jsonl"]

def convert_split(split_file):
    in_path = os.path.join(data_root, split_file)
    out_path = os.path.join(data_root, split_file.replace(".jsonl", "_alpaca.jsonl"))

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            msgs = row["messages"]

            system_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            alpaca_row = {
                "instruction": (
                    system_msg.strip()
                    + " Task: Classify the following contract clause into exactly one LEDGAR category."
                ),
                "input": user_msg.strip(),
                "output": assistant_msg.strip(),
            }

            fout.write(json.dumps(alpaca_row, ensure_ascii=False) + "\n")

    print(f"✅ Converted {in_path} → {out_path}")

if __name__ == "__main__":
    for split in splits:
        in_file = os.path.join(data_root, split)
        if os.path.exists(in_file):
            convert_split(split)
        else:
            print(f"⚠️ {in_file} not found, skipping.")
