import tiktoken 

enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "Madhav is building AI agents"
    "मैं AI सीख रहा हूं"
    "def train_model(X,Y):",
    "https://www.google.com"
    "I love you",
    " I love you I love you I love you"
]

for text in texts:
    token = enc.encode(text)
    print(f"Text: {text}")
    print(f"Token count: {len(token)}")
    print(f"Raw tokens:{token}")

    print("----")
