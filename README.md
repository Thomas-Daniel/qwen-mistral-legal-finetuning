## PFET: Finetuning Qwen 7B and Mistral 7B on Legal Datasets (LEDGAR, EURLEX)

This repository contains scripts to finetune and evaluate Qwen2-7B-Instruct and Mistral-7B-Instruct on two legal NLP datasets: `LEDGAR` and `EURLEX`. It includes both full finetuning and parameter-efficient finetuning (QLoRA/LoRA), plus evaluation utilities and dataset preparation scripts.

### Highlights
- Full finetuning for Qwen and Mistral using TRL `SFTTrainer`
- QLoRA/LoRA finetuning options
- Ready-to-run evaluation scripts with accuracy and macro-F1
- Reproducible dataset preparation for LEDGAR and EURLEX

### Repository layout
- `scripts/train/`: training scripts for full finetuning and PFET.
- `scripts/eval/`: evaluation scripts both on 500 and 1000 examples.
- `scripts/data/`: dataset preparation .
- `data/processed/...`: prepared datasets for LEDGAR and EURLEX (chat and classification formats).

### Requirements
See `requirements.txt`. Python 3.10+ recommended. Optional: FlashAttention-2 can accelerate training/inference on supported GPUs.

To use FlashAttention-2, install it separately (optional):
```bash
pip install flash-attn --no-build-isolation
```

You can use Accelerate or DeepSpeed to speed up training/inference and manage multi-GPU setups:
- Accelerate launcher (example): `accelerate launch scripts/train/train_alpaca_fullft.py ...`
- DeepSpeed launcher (example): `deepspeed --num_gpus=2 scripts/train/train_alpaca_fullft.py ...`

### Quickstart
1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Full finetuning example (Qwen2-7B on LEDGAR Alpaca-style)
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

3) Evaluate a finetuned checkpoint (Mistral example on LEDGAR)
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


### Results (LEDGAR)

Both models were trained on 5k examples due to compute restriction.

| Model                          | Setting       | Accuracy | Macro-F1 | Samples | Notes                 |
|--------------------------------|---------------|----------|----------|---------|-----------------------|
| Qwen2-7B-Instruct             | Base model    | 0.4880   | 0.3931   | 500     | attn: flash2          |
| Qwen2-7B-Instruct             | Full FT       | 0.6640   | 0.5317   | 500     | Alpaca-style          |
| Qwen2-7B-Instruct             | Full FT       | 0.6620   | 0.5579   | 1000    | Alpaca-style          |
| Mistral-7B-Instruct-v0.3      | Base model    | 0.5600   | 0.4859   | 500     | attn: flash2          |
| Mistral-7B-Instruct-v0.3      | Full FT       | 0.7560   | 0.6420   | 500     | Alpaca-style          |
| Mistral-7B-Instruct-v0.3      | Full FT       | 0.7600   | 0.6276   | 1000    | Alpaca-style          |

Note: EURLEX experiments and QLoRA/DoRA training and evaluation will be added when additional free compute is available on GCP/AWS.

### Why finetune LEDGAR with small models, and where to apply it?

- Problem fit: LEDGAR is a clause/section-level legal classification dataset derived from SEC filings. Tasks like clause tagging, contract triage, and routing are highly structured, making them ideal for targeted finetuning rather than generic prompting.
- Efficiency: Models in the 7B range (Qwen2-7B, Mistral-7B) can be finetuned to reach strong accuracy/macro-F1 on narrow legal tasks while remaining deployable on single-GPU or CPU-backed endpoints.
- Cost and latency: Smaller models reduce inference cost and latency, enabling near real-time workflows (e.g., contract clause auto-labeling in a document management system) without large GPU clusters.
- Privacy and control: On-prem or VPC deployment becomes feasible with 7B models, avoiding data sharing with third-party APIs.

Practical applications in legal workflows
- Contract analytics: auto-classify clauses (e.g., indemnification, termination) to accelerate review and redlining.
- Document routing: route agreements or filings to the right team based on detected provisions and risk profiles.
- Compliance monitoring: tag sections relevant to specific regulations for downstream checks.
- Retrieval augmentation: use predicted labels as metadata to improve precision/recall in enterprise legal search.

Integration pattern
- Ingestion: split contracts into logical sections; normalize text.
- Inference: run the finetuned 7B model to assign LEDGAR-style labels; attach confidences.
- Actions: drive review queues, apply policy checklists, or trigger template suggestions.
- Feedback loop: capture corrections from lawyers to periodically refresh the finetune.

### Notes
- For large models, ensure you have sufficient GPU memory. BF16 is enabled for speed and stability on supported hardware. Set `--flash_attn` if FlashAttention-2 is available.
- Data paths assume the processed datasets are in `./data/processed/<dataset>/...`. Adjust paths as needed if you store datasets elsewhere.

Models weights can be found on HuggingFace: 
https://huggingface.co/nobodytries/Qwen7B_Ledgar
https://huggingface.co/nobodytries/Mistral7B_LEDGAR

### License
This project is released under the MIT License. See `LICENSE` for details.