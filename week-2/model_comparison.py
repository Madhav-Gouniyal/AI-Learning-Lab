from transformers import pipeline
from groq import Groq
import json

client = Groq(api_key="your-groq-key-here")

inquiry = "I want to join your UPSC course, I have been preparing for 1 year and my budget is around 30000 rupees"

# approach 1 - sentiment classifier
sentiment = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result1 = sentiment(inquiry)
print("APPROACH 1 - SENTIMENT:")
print(f"  Label: {result1[0]['label']}, Score: {result1[0]['score']:.3f}")
print()

# approach 2 - zero shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["hot lead", "warm lead", "cold lead"]
result2 = classifier(inquiry, labels)
print("APPROACH 2 - ZERO SHOT:")
for label, score in zip(result2['labels'], result2['scores']):
    print(f"  {label}: {score:.3f}")
print()

# approach 3 - LLM classification
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": """You are a lead qualification expert.
Classify the inquiry as hot/warm/cold lead and explain why in one sentence.
Return only JSON: {"classification": "hot/warm/cold", "reason": "one sentence"}"""},
        {"role": "user", "content": inquiry}
    ]
)
result3 = json.loads(response.choices[0].message.content)
print("APPROACH 3 - LLM:")
print(f"  Classification: {result3['classification']}")
print(f"  Reason: {result3['reason']}")
print()

print("--- COMPARISON ---")
print(f"Same inquiry, 3 different approaches, 3 different depths.")
print(f"Approach 1 tells you: positive or negative")
print(f"Approach 2 tells you: which category")
print(f"Approach 3 tells you: why, with reasoning")
