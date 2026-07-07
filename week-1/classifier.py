from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "This coaching institute is amazing...",
    "Complete waste of money..."
]

for text in texts:
    result = classifier(text)

    label = result[0]['label']
    score = result[0]['score']

    print(f"Text: {text}")
    print(f"Sentiment: {label} (confidence: {score:.3f})\n")
