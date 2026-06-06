import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import fitz

load_dotenv()

PDF_PATH = "OceanofPDF.com_AI_Integration_in_Software_Development_-_Abhinav_Krishna.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))


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
        # chunk size = 500
        # overlap = 50
        # chunk1: 0 to 500
        # chunk2: 450 to 950
    return chunks


# process in batch
def embed_text(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return [item.embedding for item in response.data]


def index_pdf(pdf_path):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        chroma_client.delete_collection("pdf_rag")
    except:
        pass
    collection = chroma_client.create_collection("pdf_rag")
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []
    all_metadata = []
    all_ids = []
    chunk_index = 0

    for page_num, page_text in pages:
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append(chunk)
            all_metadata.append(
                {
                    "source": os.path.basename(pdf_path),
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
    print(f"\n Done! Indexed {len(all_chunks)} chunks from {pdf_path}")
    return collection


# retrieval pipeline
def ask(collection, question, n_results=3):
    question_embedding = embed_text(question)[0]
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk, meta, dist in zip(chunks, metadata, distances):
        print(f" [Page {meta['page']}, score {dist:.3f}, {chunk[:100]}]")

    context = "\n\n---\n\n".join(chunks)
    prompt = f""" Answer the question using ONLY the context below.
        Always mention which page the answer came from.
        If the answer is not in the context, say "I do not know."
        Context:
        {context}
    Question: {question}
    Answer:"""

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role":"user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"Answer  : {answer}")

if __name__ == "__main__":
    collection = index_pdf(PDF_PATH)
    ask(collection, 'how is Artificial intelligence being used in software development')
    ask(collection,'what are the challenges mentioned')
