import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

documents = [
    "Before committing your code to production, ensure all quality checks have been met and explicitly approved by the QA team.",
    "All code reviews must be approved by at least one Senior Developer, or a minimum of two Intermediate Developers, before merging.",
    "All newly developed features and bug fixes must include corresponding unit tests, maintaining a minimum project code coverage of 80%.",
    "Code cannot be merged into the main or release branches unless all automated Continuous Integration (CI) pipelines pass successfully.",
    "Commits containing hardcoded secrets, passwords, or API keys are strictly prohibited and will be automatically rejected and flagged for security review.",
    "Any introduction of a new third-party library or package must pass an automated security vulnerability scan before integration.",
    "All working branches must follow the standard naming convention (e.g., feature/[Ticket-ID]-[brief-description]) to ensure traceability to project management tools.",
    "Any modifications to external-facing APIs must be accompanied by immediate updates to the technical documentation (e.g., Swagger/OpenAPI).",
    "Emergency patches deployed directly to production to resolve critical incidents must undergo a formal retrospective review and back-merge within 48 hours of deployment."
]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="first_rag")

print("Embedding storing documents")

for i, doc in enumerate(documents):
    response = client.embeddings.create(
        input=doc,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[f"doc_{i}"]

    )
    print(f"Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f},...]")
    print(f"Length   : {len(embedding)} numbers\n")

print(f"Stored {len(documents)} documents in ChromaDB\n")
question = "What is the naming convention for branches?"

question_embedding = client.embeddings.create(
    input=question,
    model='text-embedding-3-small'
).data[0].embedding

#Semantic search
resuts = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

retrieved_chunks = resuts['documents'][0]
print("Received chunks:")
for chunk in retrieved_chunks:
    print(f" -> {chunk}")

print()

context = "\n".join(retrieved_chunks)
prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below
            If the answer is not in the context, say "I do not know."
            Context:
            {context}
            Question: {question}
            Answer:
        """

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{"role":"user", "content": prompt}]
)

answer = response.choices[0].message.content
print(f"Question: {question}")
print(f"Answer  : {answer}")