import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

sentences = [
    "The software was not accepted in the market becuase it had a lot of bugs",
    "The market was not happy with the software we provided",
    "The monthly earnings report is due"
]

embeddings = []

for sentence in sentences:
    response = client.embeddings.create(
        input=sentence,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding
    embeddings.append(embedding)
    print(f"Sentence : {sentence}")
    print(f"Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f},...]")
    print(f"Length   : {len(embedding)} numbers\n")


# high score = more related
def dot_product(a,b):
    return sum(x * y for x, y in zip(a,b))

sim_1_2 = dot_product(embeddings[0], embeddings[1])
sim_1_3= dot_product(embeddings[0], embeddings[2])

print("-" * 50)
print(f"Similarity: sentence 1 vs 2 {sim_1_2:.4f}")
print(f"Similarity: sentence 1 vs 3 {sim_1_3:.4f}")
print()
print("Similar sentences score HIGHER")