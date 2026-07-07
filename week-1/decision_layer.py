from transformers import pipeline

fake_news_classifier = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification"
)

def classify_and_decide(article_text, downvotes=0, upvotes=0):
    result = fake_news_classifier(article_text)
    ml_score = result[0]['score'] if result[0]['label'] == 'FAKE' else 1 - result[0]['score']

    total_votes = upvotes + downvotes
    community_score = downvotes / total_votes if total_votes > 0 else 0.5

    composite = (ml_score * 0.7) + (community_score * 0.3)

    if composite > 0.75:
        decision = "LIKELY FAKE"
    elif composite > 0.5:
        decision = "DISPUTED"
    else:
        decision = "LIKELY REAL"

    return {
        "ml_score": round(ml_score, 3),
        "community_score": round(community_score, 3),
        "composite_score": round(composite, 3),
        "decision": decision
    }

print(classify_and_decide("Government puts 5G chips in vaccines", downvotes=47, upvotes=3))
print(classify_and_decide("RBI raises interest rates by 0.25 percent", downvotes=2, upvotes=38))
print(classify_and_decide("Scientists discover water on Mars", downvotes=0, upvotes=0))
