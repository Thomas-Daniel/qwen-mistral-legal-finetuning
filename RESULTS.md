## Key Results and Reproduction Commands

This document summarizes portfolio-ready results and how to reproduce them.

### LEDGAR — Qwen2-7B-Instruct (Full Finetuning)
- Accuracy: 0.6640
- Macro-F1: 0.5317
- Samples: 500

Command:
```bash
python scripts/train/train_alpaca_fullft.py \
  --model_name Qwen/Qwen2-7B-Instruct \
  --data_root ./data/processed \
  --dataset ledgar \
  --output_dir ./qwen2-7b-ledgar-fullft \
  --epochs 2 \
  --batch_size 6 \
  --grad_accum 6 \
  --lr 2e-5 \
  --max_seq_len 1536 \
  --flash_attn \
  --no_checkpointing
```

Additional run:
- Accuracy: 0.6620
- Macro-F1: 0.5579
- Samples: 1000

### LEDGAR — Mistral-7B-Instruct (Full Finetuning)
- Accuracy: 0.7560
- Macro-F1: 0.6420
- Samples: 500

Command:
```bash
python scripts/eval/eval_sft_ledgar.py \
  --model_dir ./mistral7b-ledgar-fullft/checkpoint-278 \
  --base_model mistralai/Mistral-7B-Instruct-v0.3 \
  --eval_jsonl ./data/processed/ledgar/chat/validation_alpaca.jsonl \
  --labels_json ./data/processed/ledgar/labels.json \
  --out_dir ./eval_fullft_ledgar \
  --batch_size 1 \
  --max_new_tokens 24 \
  --max_seq_len 1536
```

Additional run:
- Accuracy: 0.7600
- Macro-F1: 0.6276
- Samples: 1000

### Notes
- All results were computed with `attn_implementation=flash2` when available.
- Ensure datasets from `data/processed/...` are present as expected.

