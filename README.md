# RAG Basics — Learning Repository

A hands-on learning repository for building **Retrieval-Augmented Generation (RAG)** pipelines from scratch using OpenAI and ChromaDB — no high-level frameworks, just raw fundamentals.

---

## Project Structure

```
rag/
└── basics/
    ├── 001_basics/
    │   ├── embedding.py        # Cosine similarity & embedding exploration
    │   └── rag.py              # Vanilla RAG with hardcoded documents
    ├── 002_basics/
    │   ├── chroma_db/          # Persistent ChromaDB vector store
    │   └── rag_pdf.py          # Full PDF ingestion & RAG pipeline
    ├── 003_api/
    │   ├── chroma_db/          # Persistent ChromaDB vector store
    │   ├── backend.py          # FastAPI backend (ingest + ask endpoints)
    │   └── frontend.py         # Streamlit UI (upload PDF, chat interface)
    ├── .env                    # API keys (never commit this)
    ├── .env.example            # Template for environment variables
    ├── .gitignore              # Ignores .venv, .env, *.pdf, __pycache__, etc.
    ├── .python-version         # Python version pin
    ├── main.py                 # Starts backend and frontend together
    ├── pyproject.toml          # Project dependencies
    ├── README.md               # This file
    └── uv.lock                 # Lockfile (uv package manager)
```

---

## Modules

### `001_basics` — Foundations

#### `embedding.py`
Explores what embeddings are and how semantic similarity works.
- Generates embeddings using OpenAI `text-embedding-3-small`
- Computes cosine similarity between sentences
- Demonstrates that similar sentences score higher

#### `rag.py` — Vanilla RAG
A minimal RAG pipeline over a hardcoded list of documents.
- Embeds 9 documents about software development coding rules
- Stores them in an in-memory ChromaDB collection
- Embeds a user question and retrieves the top-2 most relevant chunks
- Passes retrieved context to `gpt-4o-mini` for a grounded answer

---

### `002_basics` — PDF RAG Pipeline

#### `rag_pdf.py` — Full PDF Ingestion Pipeline
A complete RAG pipeline that ingests a real PDF book and answers questions about it.

| Stage | Description |
|---|---|
| Extract | Reads all pages from the PDF using PyMuPDF (fitz) |
| Chunk | Splits page text into 500-char chunks with 50-char overlap |
| Embed | Batch embeds all chunks using `text-embedding-3-small` |
| Store | Persists chunks in ChromaDB (`./chroma_db`) |
| Retrieve | Semantic search returns top-k most relevant chunks |
| Generate | `gpt-4o-mini` answers using only retrieved context, with page citations |

---

### `003_api` — Full-Stack RAG App

A production-style RAG app split into a FastAPI backend and a Streamlit frontend.

#### `backend.py` — FastAPI REST API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check, returns total chunks in DB |
| `/ingest` | POST | Upload a PDF, chunk, embed, and store in ChromaDB |
| `/ask` | POST | Ask a question, returns answer and source citations |
| `/documents` | GET | List all ingested PDF filenames |

#### `frontend.py` — Streamlit Chat UI
- Sidebar for uploading and ingesting PDFs
- Live chunk count from the backend
- Document selector to filter queries by specific PDF
- Chat interface with full message history
- Expandable source citations per answer (filename, page, similarity score)

---

## Setup

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

---

## Running

### `001_basics` — Embeddings and Vanilla RAG
```bash
uv run 001_basics/embedding.py
uv run 001_basics/rag.py
```

### `002_basics` — PDF RAG Pipeline
```bash
uv run 002_basics/rag_pdf.py
```

### `003_api` — Full-Stack App

**Option 1 — Start both servers with one command (recommended):**
```bash
uv run main.py
# Backend  running on http://localhost:8000
# Frontend running on http://localhost:8501
```

**Option 2 — Start servers manually in two terminals:**

Terminal 1:
```bash
cd 003_api
uvicorn backend:app --reload
```

Terminal 2:
```bash
cd 003_api
streamlit run frontend.py
```

Then open http://localhost:8501, upload a PDF from the sidebar, and start chatting.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `openai` | Embeddings (`text-embedding-3-small`) and LLM (`gpt-4o-mini`) |
| `chromadb` | Local vector store (in-memory and persistent) |
| `pymupdf (fitz)` | PDF text extraction |
| `fastapi` | REST API backend |
| `streamlit` | Chat UI frontend |
| `uvicorn` | ASGI server for FastAPI |
| `python-dotenv` | Environment variable management |
| `uv` | Fast Python package manager |

---

## Concepts Covered

- What embeddings are and how they encode meaning
- Cosine similarity for semantic comparison
- Fixed-size chunking with overlap
- Batch embedding for efficiency
- Persistent vector storage with ChromaDB
- Top-k semantic retrieval
- Prompt construction with retrieved context
- Grounded LLM answering with source citations
- Multi-document isolation with metadata filtering
- REST API design with FastAPI
- Chat UI with session state and source attribution

---

## Notes

This is a **vanilla / naive RAG** implementation — no LangChain, no LlamaIndex, no abstractions. Everything is built from first principles to develop a deep understanding of how RAG works before moving to higher-level frameworks.