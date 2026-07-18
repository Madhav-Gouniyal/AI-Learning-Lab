from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from supabase import create_client
import os

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SECRET_KEY")
supabase = create_client(supabase_url, supabase_key)

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

def score_lead(message):
    hot_words = ["join", "admission", "enroll", "fees", "budget", "start", "when", "book"]
    warm_words = ["interested", "tell me", "how", "course", "batch", "timing"]
    msg = message.lower()
    if any(w in msg for w in hot_words):
        return 0.8
    elif any(w in msg for w in warm_words):
        return 0.5
    return 0.2

def get_conversation_history(sender):
    try:
        result = supabase.table("conversations").select("*").eq("sender", sender).order("created_at", desc=True).limit(10).execute()
        messages = []
        for row in reversed(result.data):
            messages.append({"role": row["role"], "content": row["content"]})
        return messages
    except Exception as e:
        print(f"GET HISTORY ERROR: {e}")
        return []

def save_message(sender, role, content):
    try:
        supabase.table("conversations").insert({
            "sender": sender,
            "role": role,
            "content": content
        }).execute()
        print(f"SAVED: {role}")
    except Exception as e:
        print(f"SAVE MESSAGE ERROR: {e}")

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
            {"role": "system", "content": "You are a helpful assistant for StudyPeak UPSC coaching institute. Answer using ONLY the