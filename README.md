# Fine-Tuning BioMistral 7B for Medical Domain

This repository contains a single notebook for fine-tuning the [BioMistral 7B](https://huggingface.co/medalpaca/BioMistral-7B) model for medical chatbot applications.

## 🔬 About

- Model: BioMistral 7B
- Task: Medical Q&A / chatbot dialogue generation
- Method: [State your method here, e.g., LoRA / full fine-tuning]
- Notebook: `biomistral_finetune_medchat.ipynb`

## 📁 Contents

- `biomistral_finetune_medchat.ipynb`: Main notebook with all steps — data loading, tokenizer setup, fine-tuning loop, and inference.

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
Or install manually:
```bash
transformers
peft
accelerate
bitsandbytes
datasets
trl (if using DPO or SFTTrainer)
```

🚀 Run on Colab
You can open the notebook directly in Google Colab if you'd like to run it with free GPU.

📜 License
[MIT License]
