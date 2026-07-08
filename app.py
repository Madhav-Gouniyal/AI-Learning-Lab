from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
    query = data.get('question', '').lower()

    relevant = [chunk for chunk in knowledge_base
                if any(word in chunk.lower() for word in query.split())]

    if not relevant:
        relevant = knowledge_base[:3]

    context = "\n".join(relevant[:3])

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
            {"role": "system", "content": f"You are a WhatsApp assistant for StudyPeak UPSC coaching. Student name: {sender}. Qualify their interest in under 3 sentences."},
            {"role": "user", "content": message}
        ]
    )
    reply = response.choices[0].message.content
    return jsonify({"reply": reply, "sender": sender})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    