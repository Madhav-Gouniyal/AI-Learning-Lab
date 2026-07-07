from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from groq import Groq
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentiment_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

client = Groq(api_key="your-groq-key-here")

ideal_customer = "serious UPSC aspirant with budget ready to start immediately"

def score_lead(inquiry_text):
    # similarity between inquiry and ideal customer
    inquiry_embedding = model.encode([inquiry_text])
    ideal_embedding = model.encode([ideal_customer])
    similarity = cosine_similarity(inquiry_embedding, ideal_embedding)[0][0]

    # sentiment/enthusiasm score
    sentiment = sentiment_classifier(inquiry_text)
    enthusiasm = sentiment[0]['score'] if sentiment[0]['label'] == 'POSITIVE' else 1 - sentiment[0]['score']

    # combined lead score
    lead_score = (similarity * 0.6) + (enthusiasm * 0.4)

    # generate followup message
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are a lead follow up assistant.
Write a personalized follow up message in 2 sentences based on the lead score.
If score > 0.6: warm and urgent. If score < 0.6: gentle and informational."""},
            {"role": "user", "content": f"Lead inquiry: {inquiry_text}\nLead score: {lead_score:.2f}"}
        ]
    )

    followup = response.choices[0].message.content

    print(f"\nInquiry: {inquiry_text}")
    print(f"Lead Score: {lead_score:.2f}")
    print(f"Follow-up: {followup}")

score_lead("I have been preparing for 2 years, serious about joining, budget ready, want to start this month")
score_lead("just browsing, no budget, maybe someday")
score_lead("interested in UPSC course, want to know more details")
