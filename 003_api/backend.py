import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import fitz
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile

load_dotenv()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

app = FastAPI(
    title='RAG API'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection():
    try:
        return chroma_client.get_collection('rag_collection')
    except Exception as e:
        print(e)
        return chroma_client.create_collection('rag_collection')


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text = text.strip()
        if text:
            pages.append((page_num + 1, text))
    doc.close()
    print(f"Extracted text from {len(pages)} pages")
    return pages

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_text(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return [item.embedding for item in response.data]

@app.post('/ingest')
async def ingest(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    collection = get_or_create_collection()
    pages = extract_text_from_pdf(tmp_path)
    all_chunks = []
    all_metadata = []
    all_ids = []
    chunk_index = collection.count()

    # ── Store filename as metadata chunk ──
    meta_text = f"Document filename and title: {file.filename}"
    meta_embedding = embed_text(meta_text)[0]
    collection.add(
        documents=[meta_text],
        embeddings=[meta_embedding],
        metadatas=[{"source": file.filename, "page": 0, "chunk": chunk_index}],
        ids=[f"meta_{file.filename}"]
    )
    chunk_index += 1

    # ── Store first 3 pages as a combined intro chunk ──
    intro_pages = [text for _, text in pages[:3]]
    intro_text = "Book introduction, title page and author information:\n\n" + "\n\n".join(intro_pages)
    intro_embedding = embed_text(intro_text)[0]
    collection.add(
        documents=[intro_text],
        embeddings=[intro_embedding],
        metadatas=[{"source": file.filename, "page": 1, "chunk": chunk_index}],
        ids=[f"intro_{file.filename}"]
    )
    chunk_index += 1

    for page_num, page_text in pages:
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append(chunk)
            all_metadata.append(
                {
                    "source": file.filename,
                    "page": page_num,
                    "chunk": chunk_index,
                }
            )
            all_ids.append(f"chunk_{chunk_index}")
            chunk_index += 1

    BATCH_SIZE = 100
    all_embeddings = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        embeddings = embed_text(batch)
        all_embeddings.extend(embeddings)
        print(f"Embedded batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids,
    )
    return {
        'message': f'Ingested {file.filename} successfully',
        'chunks_added': len(all_chunks),
        'total_chunks': collection.count()
    }

class AskRequest(BaseModel):
    question: str
    n_results: int = 3
    source: str | None = None

@app.get('/documents')
async def list_documents():
    collection = get_or_create_collection()
    result = collection.get(include=['metadatas'])
    sources = list(set(
        m['source'] for m in result['metadatas']
    ))
    return {'sources': sources}

@app.post('/ask')
async def ask(request: AskRequest):
    question_embedding = embed_text(request.question)[0]
    collection = get_or_create_collection()

    if collection.count() == 0:
        return {'error': 'No documents ingested yet'}

    # ── Apply source filter if provided ──
    where = {"source": request.source} if request.source else None

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=request.n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    sources = [
        {
            "source": meta["source"],
            "page": meta["page"],
            "score": round(dist, 3)
        }
        for meta, dist in zip(metadatas, distances)
    ]

    context_parts = []
    for chunk, meta in zip(chunks, metadatas):
        context_parts.append(
            f"[Source: {meta['source']}, Page {meta['page']}]\n{chunk}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Answer the question using ONLY the context below.
Always mention which page the answer came from.
If the answer is not in the context, say "I do not know."

Context:
{context}

Question: {request.question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "question": request.question
    }


@app.get('/')
async def root():
    collection = get_or_create_collection()
    return {
        'status': 'ok',
        'total_chunks': collection.count()
    }