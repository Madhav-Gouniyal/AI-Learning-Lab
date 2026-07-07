from transformers import pipeline
from groq import Groq

fake_news_classifier = pipeline(
    "text-classification",
    model = "hamza/roberta-fake-news-classification"

)
client = Groq(api_key="your-groq-key")

def full_truthchain(article_text, downvotes =0, upvotes =0):
    result = fake_news_classifier(article_text)
    ml_score = result[0]['score'] if result[0]['label'] == 'FAKE' else 1 - result[0]['score']

    total_votes = upvotes + downvotes
    community_score = downvotes/ total_votes if total_votes > 0 else 0.5
    composite = (ml_score* 0.7) + (community_score*0.3)

    if composite > 0.75:
        descision = "LIKELY FAKE"

    elif composite > 0.5:
        decision = "DISPUTED"
    else:
        decision = "LIKELY REAL"


response = client.chat.completions.create(
    model = "llama-3.3-40b-versatile",
    messages =[
        {"role": "system", "content": """You are a fact-checking assistant.
       ased on the article and scores provided, write exactly 2 sentences explaining why this article was flagged. Be specific and neutral."""},
       {"role": "user", "content": f"""Article:{article_text}}
     ML fake score: {ml_score}
     Community downvote ratio: {community_score}
     Final decision:{decision}

     Explain why this article was flagged. """}
    ]
)

explanation = response.choices[0].message.content

print(f"\nArticle:{article_text}")
print(f"Decision:{decision}")
print(f"Explanation:{explanation}")

full_truthchain("Government puts 5G chips in vaccines", downvotes = 47, upvotes = 3)
full_truthchain("RBI raises interest rates by 0.25 percent", downvotes =2, upvotes = 38)
full_truthchain("Scientists discover water on Mars", downvotes = 0, upvotes = 0)
