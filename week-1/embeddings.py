from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I want to learn machine learning",
    "I want to study artificial intelligence",
    "I want to eat pizza",
    "UPSC coaching classes in Delhi",
    "IAS exam preparation institute",
    "The stock market crashed today"
]

embeddings = model.encode(sentences)
print(f"Vector size: {len(embeddings[0])}\n")

similarity_matrix = cosine_similarity(embeddings)

for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        if i < j:
            score = similarity_matrix[i][j]
            print(f"{score:.2f} | {s1} vs {s2}")
