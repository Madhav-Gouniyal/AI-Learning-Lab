from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.environ.get("your-groq-key-here"))

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

@app.route('/')
def home():
    return jsonify({"status": "StudyPeak AI Agent is running"})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('question', '')

    query_embedding = model.encode([query])
    kb_embeddings = model.encode(knowledge_base)
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:3]
    context = "\n".join([knowledge_base[idx] for idx in top_indices])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for StudyPeak UPSC coaching institute. Answer using ONLY the provided context. Be concise and friendly."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    answer = response.choices[0].message.content
    return jsonify({"answer": answer, "query": query})

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    message = data.get('message', '')
    sender = data.get('sender', 'Student')

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You are a WhatsApp assistant for StudyPeak UPSC coaching. The student's name is {sender}. Qualify their interest naturally in under 3 sentences."},
            {"role": "user", "content": message}
        ]
    )
    reply = response.choices[0].message.content
    return jsonify({"reply": reply, "sender": sender})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
