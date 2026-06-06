# 🧠 RAG Basics — Learning Repository

A hands-on learning repository for building **Retrieval-Augmented Generation (RAG)** pipelines from scratch using OpenAI and ChromaDB — no high-level frameworks, just raw fundamentals.

---

## 📁 Project Structure

```
rag/
└── basics/
    ├── 001_basics/
    │   ├── embedding.py        # Cosine similarity & embedding exploration
    │   └── rag.py              # Vanilla RAG with hardcoded documents
    ├── 002_basics/
    │   ├── chroma_db/          # Persistent ChromaDB vector store
    │   ├── rag_pdf.py          # Full PDF ingestion & RAG pipeline
    │   └── OceanofPDF.com_AI_Integration_in_Software_Development_-_Abhinav_Krishna.pdf
    ├── .env                    # API keys (never commit this)
    ├── .env.example            # Template for environment variables
    ├── .gitignore              # Ignores .venv, .env, *.pdf, __pycache__, etc.
    ├── .python-version         # Python version pin
    ├── main.py                 # Entry point
    ├── pyproject.toml          # Project dependencies
    ├── README.md               # This file
    └── uv.lock                 # Lockfile (uv package manager)
```

---

## 🔬 Modules

### `001_basics` — Foundations

#### `embedding.py`
Explores what embeddings are and how semantic similarity works.
- Generates embeddings using OpenAI `text-embedding-3-small`
- Computes **cosine similarity** between sentences
- Demonstrates that similar sentences score higher

#### `rag.py` — Vanilla RAG
A minimal RAG pipeline over a hardcoded list of documents.
- Embeds 9 documents about software development coding rules
- Stores them in an **in-memory ChromaDB** collection
- Embeds a user question and retrieves the top-2 most relevant chunks
- Passes retrieved context to `gpt-4o-mini` for a grounded answer

---

### `002_basics` — PDF RAG Pipeline

#### `rag_pdf.py` — Full PDF Ingestion Pipeline
A complete RAG pipeline that ingests a real PDF book and answers questions about it.

**Pipeline stages:**

| Stage | Description |
|---|---|
| **Extract** | Reads all 313 pages from the PDF using `PyMuPDF (fitz)` |
| **Chunk** | Splits page text into 500-char chunks with 50-char overlap |
| **Embed** | Batch embeds all chunks using `text-embedding-3-small` |
| **Store** | Persists 1,135 chunks in ChromaDB (`./chroma_db`) |
| **Retrieve** | Semantic search returns top-k most relevant chunks |
| **Generate** | `gpt-4o-mini` answers using only retrieved context, with page citations |

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd rag/basics

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPEN_AI_KEY=sk-...
```

### Running

```bash
# Run embeddings exploration
uv run 001_basics/embedding.py

# Run vanilla RAG
uv run 001_basics/rag.py

# Run PDF RAG pipeline
uv run 002_basics/rag_pdf.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `openai` | Embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`) |
| `chromadb` | Local vector store (in-memory & persistent) |
| `pymupdf (fitz)` | PDF text extraction |
| `python-dotenv` | Environment variable management |
| `uv` | Fast Python package manager |

---

## 📚 Concepts Covered

- ✅ What embeddings are and how they encode meaning
- ✅ Cosine similarity for semantic comparison
- ✅ Fixed-size chunking with overlap
- ✅ Batch embedding for efficiency
- ✅ Persistent vector storage with ChromaDB
- ✅ Top-k semantic retrieval
- ✅ Prompt construction with retrieved context
- ✅ Grounded LLM answering with source citations

---

## 🗺️ What's Next

- [ ] Semantic / sentence-aware chunking
- [ ] Metadata filtering in retrieval
- [ ] Hybrid search (semantic + keyword BM25)
- [ ] Re-ranking retrieved chunks
- [ ] RAG evaluation with RAGAS
- [ ] Agentic / multi-step retrieval

---

## 📝 Notes

> This is a **vanilla / naive RAG** implementation — no LangChain, no LlamaIndex, no abstractions. Everything is built from first principles to develop a deep understanding of how RAG works before moving to higher-level frameworks.