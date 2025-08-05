# 🧠 Fine-Tuning BioMistral 7B for Medical Chatbot (QLoRA + Unsloth on Colab)

This project demonstrates how to fine-tune [BioMistral 7B](https://huggingface.co/BioMistral/BioMistral-7B) for medical chatbot applications using [QLoRA](https://arxiv.org/abs/2305.14314) and [Unsloth](https://github.com/unslothai/unsloth) in a single Colab notebook.

It leverages the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset to train the model using supervised fine-tuning with a reasoning-augmented format.

---

## 📁 File Structure

```bash
biomistral-qlora-colab-finetune/
├── BioMistral_Unsloth.ipynb # Main Colab notebook for QLoRA fine-tuning
├── biomistral_chat_format.jsonl # Preprocessed dataset (optional, generated)
├── outputs/ # Directory for saving model artifacts
└── README.md # Project documentation
```


---

## 🚀 What This Notebook Covers

- ✅ Automatically detects GPU architecture (Ampere, Turing, etc.)
- ✅ Installs required libraries using `uv` and pip
- ✅ Loads BioMistral 7B in 4-bit with `Unsloth`
- ✅ Adds LoRA adapters using QLoRA
- ✅ Prepares dataset in ChatML format
- ✅ Fine-tunes using `trl.SFTTrainer`
- ✅ Runs inference on clinical prompts
- ✅ Saves and optionally converts to GGUF format
- ✅ (Optional) Pushes model to Hugging Face Hub

---

## 📚 Dataset Used

- **Name**: [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **Format**: Converted to ChatML-style messages with reasoning + response

---

## 📦 Required Libraries

The notebook installs these automatically in Colab:

```bash
uv pip install "unsloth[colab_ampere] @ git+https://github.com/unslothai/unsloth.git"
uv pip install "git+https://github.com/huggingface/transformers.git"
uv pip install trl datasets
```

---

🧪 Run in Google Colab
You can open and run this notebook on Colab with GPU (T4, A100, etc.):

```bash
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What’s the diagnosis for persistent cough + weight loss?"}], tokenize=False, add_generation_prompt=True)

output = pipe(prompt, max_new_tokens=512)
print(output[0]['generated_text'])
```

---

## 💾 Saving & Exporting
Saves with unsloth_save_model

Can export to GGUF format using colab_quantize_to_gguf

Optional Hugging Face push via model.push_to_hub_gguf(...)

---

### 📜 License
This project is open-sourced under the MIT License. Use it responsibly, especially when handling medical data.

---

## 🙌 Credits

BioMistral 7B
Unsloth
Transformers
FreedomIntelligence Dataset

--- 

## 🔗 Related
🔬 QLoRA Paper
🐍 Unsloth on GitHub
🤗 Hugging Face Transformers

---

Would you also like a `requirements.txt` just in case someone wants to run this **locally** or on **Kaggle** instead of Colab?

Let me know and I can generate that too.
