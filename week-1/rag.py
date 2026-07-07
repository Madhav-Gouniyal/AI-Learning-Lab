from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key="your-groq-key-here")

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

def rag_answer(query):
    query_embedding = model.encode([query])
    kb_embeddings = model.encode(knowledge_base)
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:3]
    context = "\n".join([knowledge_base[idx] for idx in top_indices])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are a helpful assistant for a UPSC coaching institute.
Answer using ONLY the provided context.
If answer is not in context, say I don't have that info, please call us.
Be concise and friendly."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    print(f"\nQ: {query}")
    print(f"A: {response.choices[0].message.content}")

rag_answer("what are the fees")
rag_answer("I am from Mumbai can I still join")
rag_answer("what time are classes")
rag_answer("do you have payment plans")
    


