# 💊 Paracetamol Document QA System
### A Retrieval-Augmented Generation (RAG) Application
**Built with:** LangChain · Streamlit · Ollama · FAISS

> **100% Local · No API Keys · No Cloud · No Cost**  
> Ask intelligent questions about a Paracetamol PDF — answers grounded in the document.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [RAG Pipeline](#-rag-pipeline)
4. [Features](#-features)
5. [Prerequisites](#-prerequisites)
6. [Installation & Setup](#-installation--setup)
7. [Running the App](#-running-the-app)
8. [Usage Guide](#-usage-guide)
9. [Configuration Options](#-configuration-options)
10. [Project Structure](#-project-structure)
11. [Dependencies](#-dependencies)
12. [Troubleshooting](#-troubleshooting)
13. [Technology Stack](#-technology-stack)

---

## 📌 Project Overview

This project implements a **Document Question Answering system** using **Retrieval-Augmented Generation (RAG)**. Instead of relying on a language model's pre-trained knowledge alone, the system retrieves the most relevant sections from an uploaded PDF and uses them as context to generate accurate, document-grounded answers.

The system is built entirely on **local, open-source tools** — no paid API keys or internet connection required after initial setup.

### What it does
- Accepts a **Paracetamol PDF document** as input
- Splits it into searchable chunks and indexes them using vector embeddings
- When a user asks a question, it retrieves the **most semantically relevant chunks**
- Passes those chunks as context to a **local LLM (via Ollama)** to generate a precise answer
- Displays the answer along with the **source document chunks** used

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│         Sidebar Controls │ PDF Upload │ Q&A Interface        │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │   DOCUMENT PROCESSOR  │
          │   PyPDFLoader         │
          │   RecursiveTextSplit  │
          └───────────┬───────────┘
                      │ chunks
          ┌───────────▼───────────┐
          │   EMBEDDING ENGINE    │
          │   Ollama Embeddings   │◄──── Any installed Ollama model
          │   OR HuggingFace      │◄──── sentence-transformers (fallback)
          └───────────┬───────────┘
                      │ vectors
          ┌───────────▼───────────┐
          │   FAISS VECTOR STORE  │
          │   Similarity Search   │
          └───────────┬───────────┘
                      │ top-K chunks
          ┌───────────▼───────────┐
          │   RAG CHAIN (LCEL)    │
          │   Context + Question  │
          │   → Prompt Template   │
          │   → Ollama LLM        │
          │   → StrOutputParser   │
          └───────────┬───────────┘
                      │ answer + sources
          ┌───────────▼───────────┐
          │   STREAMLIT DISPLAY   │
          │   Answer + Chunks     │
          └───────────────────────┘
```

---

## 🔄 RAG Pipeline

The RAG pipeline follows these **7 steps**:

| Step | Component | Description |
|------|-----------|-------------|
| **1. Load** | `PyPDFLoader` | Reads and parses the uploaded PDF |
| **2. Chunk** | `RecursiveCharacterTextSplitter` | Splits text into overlapping chunks (default: 1000 tokens, 200 overlap) |
| **3. Embed** | `OllamaEmbeddings` / `HuggingFaceEmbeddings` | Converts chunks to dense vector representations |
| **4. Index** | `FAISS` | Stores vectors for fast similarity search |
| **5. Retrieve** | FAISS retriever | Fetches top-K most relevant chunks for the user query |
| **6. Generate** | `OllamaLLM` | Local LLM generates an answer using retrieved context |
| **7. Display** | Streamlit | Shows answer + source chunk expanders with page numbers |

### Prompt Design

The system uses a strict context-grounded prompt:

```
You are an expert assistant specialized in Paracetamol questions.

Instructions:
1. Use ONLY the provided context to answer the question
2. If the answer is not in the context, clearly say you don't know
3. Be accurate, clear, and concise

Context: {context}
Question: {question}
Answer:
```

This ensures the model **does not hallucinate** or rely on pre-trained knowledge outside the document.

---

## ✨ Features

- ✅ **100% Local** — Runs entirely on your machine via Ollama
- ✅ **No API Keys** — No OpenAI, Google, or Anthropic billing
- ✅ **Any Ollama Model** — Works with llama3, mistral, gemma, phi, deepseek, and more
- ✅ **Smart Embedding Fallback** — Auto-detects best available embedding model:
  - Dedicated embed model (e.g. `nomic-embed-text`) → best quality
  - Any installed chat model → good quality, no extra pull
  - HuggingFace `all-MiniLM-L6-v2` → always available, no pull needed
- ✅ **In-App Model Pulling** — Pull Ollama models directly from the sidebar
- ✅ **Live Connection Status** — Shows Ollama status and installed models in real time
- ✅ **Source Transparency** — Every answer shows the exact document chunks used
- ✅ **Configurable Pipeline** — Adjust chunk size, temperature, top-K, max tokens via sliders
- ✅ **Document Caching** — Re-indexing skipped if same file + embedding model used

---

## ⚙️ Prerequisites

### 1. Python
- Python **3.9 or higher**
- Verify: `python --version`

### 2. Ollama
Ollama is the local LLM runtime this app uses.

**Install Ollama:**
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download installer from: https://ollama.com/download
```

**Start the Ollama server:**
```bash
ollama serve
```

**Pull at least one chat model:**
```bash
ollama pull llama3          # recommended (best balance)
# OR
ollama pull mistral         # good alternative
# OR
ollama pull phi3            # lightweight (low RAM)
```

**Optional — dedicated embedding model (better retrieval quality):**
```bash
ollama pull nomic-embed-text    # best embedding model
# OR
ollama pull mxbai-embed-large   # alternative
```

> **Note:** If you don't pull an embedding model, the app automatically falls back to using your chat model for embeddings, or HuggingFace `all-MiniLM-L6-v2` — both work fine.

---

## 🛠️ Installation & Setup

### Step 1 — Clone or download the project

```bash
git clone <your-repo-url>
cd paracetamol-rag
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Start Ollama

```bash
ollama serve
```

Keep this terminal open. Ollama must be running for the app to work.

### Step 5 — Pull your model (if not done already)

```bash
ollama pull llama3
```

---

## ▶️ Running the App

```bash
streamlit run paracetamol_ollama_rag.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📖 Usage Guide

### Step 1 — Check Ollama Status
In the sidebar, verify **"✅ Ollama is running"** and your installed models are listed.

### Step 2 — Select Your Chat Model
Choose from the **Chat Model** dropdown. Installed models appear at the top.  
If a model isn't installed, click the **⬇️ Pull** button to download it from within the app.

### Step 3 — Select Embedding Source
Choose how document chunks will be embedded:
- 🦙 **Any installed Ollama model** — works immediately
- ⭐ **nomic-embed-text** — best quality (pull it from the expander below)
- 🤗 **HuggingFace all-MiniLM-L6-v2** — no pull needed, always available

### Step 4 — Upload the Paracetamol PDF
Click **"📁 Upload Paracetamol PDF"** and select your document.  
The app will chunk and embed it automatically (a spinner shows progress).

### Step 5 — Ask Questions
Type your question in the text box, e.g.:
- *"What are the side effects of Paracetamol?"*
- *"What is the recommended dosage for adults?"*
- *"Is Paracetamol safe during pregnancy?"*
- *"What are the overdose symptoms?"*

### Step 6 — Review the Answer
- The **📝 Answer** section shows the model's response
- The **📚 Source Chunks** section shows the exact PDF passages used — expand each to verify

---

## ⚙️ Configuration Options

All settings are available in the **sidebar** without touching the code:

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| **Chat Model** | First installed | Any Ollama model | LLM used for answer generation |
| **Embedding Source** | Auto-detected | Ollama / HuggingFace | Model used to vectorize chunks |
| **Temperature** | 0.5 | 0.0 – 1.0 | Creativity of responses (lower = more factual) |
| **Max Output Tokens** | 1024 | 256 – 4096 | Maximum length of generated answer |
| **Top-K Chunks** | 3 | 1 – 10 | Number of document chunks retrieved per query |
| **Chunk Size** | 1000 | 200 – 2000 | Size of text chunks in tokens |
| **Chunk Overlap** | 200 | fixed | Overlap between consecutive chunks |
| **Ollama Host URL** | localhost:11434 | any URL | Change if Ollama runs on a remote machine |

---

## 📁 Project Structure

```
paracetamol-rag/
│
├── paracetamol_ollama_rag.py   # Main application file
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .env                         # Optional: set OLLAMA_HOST here
│
└── docs/
    └── paracetamol.pdf          # Place your Paracetamol PDF here
```

---

## 📦 Dependencies

### requirements.txt

```txt
streamlit>=1.32.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-core>=0.2.0
faiss-cpu>=1.7.4
pypdf>=4.0.0
requests>=2.31.0
python-dotenv>=1.0.0
sentence-transformers>=2.7.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

### ❌ "Ollama not reachable"
```bash
# Make sure Ollama is running
ollama serve

# Check it's accessible
curl http://localhost:11434/api/tags
```

### ❌ "Model not found — try pulling it first"
```bash
# Pull the model shown in the error
ollama pull llama3

# List installed models
ollama list
```

### ❌ "Embedding failed with nomic-embed-text"
The embedding model isn't pulled. Either:
- Pull it: `ollama pull nomic-embed-text`
- **Or switch to HuggingFace** in the sidebar (no pull needed — always works)

### ❌ App is very slow on first embedding
This is normal — the first time a model processes embeddings it loads into memory. Subsequent queries are much faster.

### ❌ "PDF has no readable pages"
- Make sure the PDF isn't password-protected or image-only (scanned)
- Try a different PDF viewer to confirm the text is selectable

### ❌ High RAM usage
Switch to a smaller model:
```bash
ollama pull phi3          # ~2.3GB RAM
ollama pull llama3.2      # ~2GB RAM
ollama pull gemma2        # ~3GB RAM
```

---

## 🧰 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | Streamlit | Web interface |
| **LLM Runtime** | Ollama | Local model serving |
| **LLM Orchestration** | LangChain (LCEL) | Chain building, prompt templates |
| **Document Loader** | PyPDFLoader (LangChain) | PDF parsing |
| **Text Splitter** | RecursiveCharacterTextSplitter | Chunking |
| **Embeddings** | Ollama `/api/embeddings` | Vector generation (primary) |
| **Embeddings (fallback)** | HuggingFace sentence-transformers | Vector generation (no-pull fallback) |
| **Vector Store** | FAISS | Similarity search & indexing |
| **Output Parser** | StrOutputParser | Clean text extraction |

---

## 👨‍💻 Author Notes

- This project was built as part of a RAG systems assignment
- The system is designed to be **provider-agnostic for embeddings** — any Ollama model works
- The HuggingFace fallback ensures the app works **even with zero dedicated embedding models** installed
- All processing is local — the Paracetamol document never leaves your machine

---

*Built with ❤️ using LangChain, Streamlit, and Ollama*
