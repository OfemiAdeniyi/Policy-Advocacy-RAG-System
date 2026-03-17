# Policy-RAG: High-Resilience Document Intelligence System

A production-grade Retrieval-Augmented Generation (RAG) system engineered to ingest, process, and query massive legal and policy repositories with high accuracy and sub-second latency.

---

## 🚀 The Challenge

Legal and government archives are notoriously difficult to digitize. During development, this system successfully navigated:

- **Data Corruption:** Overcoming "broken" PDF headers and legacy HTML-disguised files that standard loaders fail to read.
- **Massive Document Density:** Handling "heavyweight" documents (5,000+ internal relationships) without crashing system memory.
- **Infrastructure Efficiency:** Optimizing for cost and performance using a hybrid local/cloud processing model.

---

## 🛠️ Technical Architecture

The system is built on a **Stateful Resilience Layer**, ensuring that data ingestion is fault-tolerant and scalable.

- **Framework:** LangChain (Classic & Core 2026 Modular Standards)
- **Inference:** Groq LPU (Language Processing Units) for near-instant response times
- **LLM:** Llama 3.1 8B (Optimized for policy nuance and legal context)
- **Vector Store:** ChromaDB with manual batch pagination to bypass SQLite transaction limits
- **Hybrid Parser:** Custom logic gate using local CPU parsing for digital PDFs and AI-powered Cloud OCR (LlamaParse) for unreadable scans

---

## ✨ Key Features

### 1. Stateful Ingestion & Resume Logic
Built-in checkpointing allows the system to track progress at a file level. If interrupted, it resumes instantly without redundant processing or API credit waste.

### 2. Manual Batch Pagination
Handles dense documents exceeding vector limits by segmenting data into safe batches, ensuring database stability.

### 3. Cost-Optimized Hybrid Pipeline
Automatically detects document types to minimize cost—using local resources for most tasks and cloud AI only when necessary.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/policy-rag.git

# Install dependencies
pip install langchain-groq langchain-huggingface langchain-chroma \
            langchain-classic llama-index-readers-llama-parse
```

---

## 🖥️ Usage

### Ingestion
Place your PDF files in the `/source_docs` folder and run:

```bash
python ingest.py
```

### Querying
Run the chat interface:

```bash
python chat.py
```

---

## 📊 Performance Metrics

- **Documents Indexed:** 1,300+
- **Inference Speed:** ~500 tokens/sec (via Groq LPU)
- **Data Integrity:** 100% recovery rate on legacy/broken PDF headers

---

## 📌 Project Highlights

- Built for **real-world policy and legal data challenges**
- Designed with **resilience, scalability, and cost-efficiency in mind**
- Suitable for **government, research, and enterprise document intelligence systems**

---

## 👨‍💻 Author

Micheal Adeniyi  
Data & AI Engineer | HealthTech Innovator  
GitHub: https://github.com/OfemiAdeniyi
