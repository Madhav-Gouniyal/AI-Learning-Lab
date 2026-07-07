from transformers import pipeline
from groq import Groq

fake_news_classifier = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification"
)

client = Groq(api_key="your-groq-key")

def truthchain(article_text, downvotes=0, upvotes=0):
    # layer 1 - ml prediction
    result = fake_news_classifier(article_text)
    ml_score = result[0]['score'] if result[0]['label'] == 'FAKE' else 1 - result[0]['score']

    # layer 2 - decision
    total_votes = upvotes + downvotes
    community_score = downvotes / total_votes if total_votes > 0 else 0.5
    composite = (ml_score * 0.7) + (community_score * 0.3)

    if composite > 0.75:
        decision = "LIKELY FAKE"
    elif composite > 0.5:
        decision = "DISPUTED"
    else:
        decision = "LIKELY REAL"

    # layer 3 - llm explanation
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a fact-checking assistant. Write a 2-sentence plain English explanation of the decision based on the scores given."},
            {"role": "user", "content": f"Article: {article_text}\nML score: {ml_score:.2f}\nCommunity downvote ratio: {community_score:.2f}\nDecision: {decision}\nExplain why."}
        ]
    )
    explanation = response.choices[0].message.content

    print(f"\nArticle: {article_text}")
    print(f"Decision: {decision} (composite: {composite:.2f})")
    print(f"Explanation: {explanation}")

truthchain("Government puts 5G chips in vaccines", downvotes=47, upvotes=3)
truthchain("RBI raises interest rates by 0.25 percent", downvotes=2, upvotes=38)
truthchain("Scientists discover new species in Amazon rainforest", downvotes=0, upvotes=0)
