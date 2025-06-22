# ❤️ Heart-Disease-QLoRA-RAG


This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:

- A **QLoRA fine-tuned** [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) language model
- The model was **fine-tuned using a custom heart disease dataset**
- **LangChain** for orchestrating retrieval and generation
- **FAISS** for vector-based document search
- **Open-source MiniLM embeddings**
- **PDF documents** as the knowledge base

---

## 🚀 Features

- 🧠 QLoRA fine-tuned LLM trained on **custom heart disease dataset**
- 🔍 PDF-based semantic search using FAISS
- 🧩 Embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- ⚡ Fast local inference with DeepSeek 1.5B

---

## 🧱 Tech Stack

| Component      | Tool/Library                                |
|----------------|---------------------------------------------|
| Language Model | `DeepSeek-R1-Distill-Qwen-1.5B` (Fine-tuned on custom heart disease data) |
| Vector DB      | `FAISS`                                     |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2`    |
| Framework      | `LangChain`, `Transformers`, `PEFT`         |
| File Loader    | `pypdf`                                     |
| Environment    | `Google Colab` or any Python environment    |

---


## 🛠️ Setup Instructions

1. Clone the repo and run in Google Colab or your local Python environment

2. Install dependencies:

```bash
# libraries for QLoRA fine-tuning
pip install -q -U transformers datasets peft accelerate bitsandbytes wandb


# libraries for RAG pipeline
pip install -U langchain transformers accelerate peft \
sentence-transformers faiss-cpu pypdf


