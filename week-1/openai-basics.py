from groq import Groq

client = Groq(api_key="your-groq-key-here")

messages = [ {"role": "system", "content": "you are a helpful assistant for StudyPeak UPSC coaching in Delhi."}]

print("Chat with StudyPeak AI(type quit to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages=messages
    )

    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})

    print(f"Bot: {assistant_message}\n")



