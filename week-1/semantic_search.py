from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

knowledge_base = [
    "Our UPSC course costs 45000 rupees for the full program",
    "Classes are held Monday to Friday from 7am to 9am",
    "We offer EMI options in 3, 6, and 12 month installments",
    "The course duration is 12 months with 400 hours of content",
    "Our success rate is 34% for prelims in the 2023 batch",
    "You can contact us at 9876543210 or visit us in Laxmi Nagar",
    "We provide free study material and test series with enrollment",
    "Online batches are available for outstation students"
]

def search(query, top_k=3):
    query_embedding = model.encode([query])
    kb_embeddings = model.encode(knowledge_base)
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    print(f"\nQuery: '{query}'")
    for idx in top_indices:
        print(f"  {similarities[idx]:.3f} — {knowledge_base[idx]}")

search("how much does the course cost")
search("can I pay in parts")
search("do you have online option")
search("what time are the classes")
