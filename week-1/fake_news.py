from transformers import pipeline

fake_news_classifier = pipeline(
    "text-classification",model = "hamzab/roberta-fake-news-classification"

)

articles = [
    "scientists discover water on mars in underground lake "
    "Governent puts 5G chips in vaccines to control population"
     "RBI raises interest rates by 0.25 percent in latest meeting",
    "Eating chocolate every day makes you immortal says new study"

]

for articles in articles:
    result = fake_news_classifier(article)
    print(f"Article: {article}")
    print(f"Classification: {result[0]['label']} (confidence:{result[0]['score']:.3f})\n")
    
