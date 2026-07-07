from transformers import pipeline

# task 1 - text generation
generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial intelligence is transforming Indian businesses by"

result = generator(prompt, max_new_tokens=50, num_return_sequences=1)
print("TEXT GENERATION:")
print(result[0]['generated_text'])
print()

# task 2 - named entity recognition
ner = pipeline("ner",
               model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple")

text = "Madhav Gouniyal from KIIT University in Ghaziabad is building AI agents for Indian businesses in Delhi NCR."

entities = ner(text)
print("NAMED ENTITY RECOGNITION:")
for entity in entities:
    print(f"  {entity['word']} -> {entity['entity_group']} (confidence: {entity['score']:.3f})")
print()

# task 3 - zero shot classification
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

inquiry = "I want to know about fees and batch timings for UPSC preparation"
labels = ["pricing inquiry", "schedule inquiry", "general inquiry", "complaint", "enrollment ready"]

result = classifier(inquiry, labels)
print("ZERO SHOT CLASSIFICATION:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.3f}")

