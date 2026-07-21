from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from supabase import create_client
import os

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
supabase = create_client(os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_SECRET_KEY'))

kb = [
    'Our UPSC course costs 45000 rupees for the full program',
    'Classes are held Monday to Friday from 7am to 9am',
    'We offer EMI options in 3, 6, and 12 month installments',
    'The course duration is 12 months with 400 hours of content',
    'Our success rate is 34 percent for prelims in the 2023 batch',
    'You can contact us at 9876543210 or visit us in Laxmi Nagar',
    'We provide free study material and test series with enrollment',
    'Online batches are available for outstation students'
]

def score_lead(message):
    hot = ['join','admission','enroll','fees','budget','start','when','book']
    warm = ['interested','tell me','how','course','batch','timing']
    m = message.lower()
    if any(w in m for w in hot): return 0.8
    if any(w in m for w in warm): return 0.5
    return 0.2

def get_history(sender):
    try:
        r = supabase.table('conversations').select('*').eq('sender', sender).order('created_at', desc=True).limit(10).execute()
        return [{'role': x['role'], 'content': x['content']} for x in reversed(r.data)]
    except Exception as e:
        print('GET HISTORY ERROR:', e)
        return []

def save_message(sender, role, content):
    try:
        supabase.table('conversations').insert({'sender': sender, 'role': role, 'content': content}).execute()
        print('SAVED:', role)
    except Exception as e:
        print('SAVE MESSAGE ERROR:', e)

@app.route('/')
def home():
    return jsonify({'status': 'StudyPeak AI Agent is running'})

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    message = data.get('message', '')
    sender = data.get('sender', 'Student')
    print('WEBHOOK HIT:', sender, message)
    history = get_history(sender)
    print('HISTORY COUNT:', len(history))
    msgs = [{'role': 'system', 'content': 'You are a WhatsApp assistant for StudyPeak UPSC coaching. Remember what the student told you before. Qualify their interest naturally in under 3 sentences.'}]
    msgs.extend(history)
    msgs.append({'role': 'user', 'content': message})
    resp = client.chat.completions.create(model='llama-3.3-70b-versatile', messages=msgs)
    reply = resp.choices[0].message.content
    save_message(sender, 'user', message)
    save_message(sender, 'assistant', reply)
    try:
        supabase.table('leads').insert({'sender': sender, 'message': message, 'reply': reply, 'lead_score': score_lead(message)}).execute()
    except Exception as e:
        print('LEADS ERROR:', e)
    return jsonify({'reply': reply, 'sender': sender})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
