from groq import Groq
import json

client = Groq(api_key="your-groq-key-here")  # Replace with your actual API key


class LeadAgent:
    def __init__(self, institute_name="StudyPeak"):
        self.institute_name = institute_name
        self.conversation = [
            {
                "role": "system",
                "content": f"""
You are a lead qualification assistant for {self.institute_name}.

Ask qualifying questions naturally.
After 3 exchanges, classify the lead and recommend an action.
Keep responses under 3 sentences, friendly and conversational.
"""
            }
        ]

    def respond(self, user_message):
        self.conversation.append(
            {"role": "user", "content": user_message}
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=self.conversation
        )

        bot_reply = response.choices[0].message.content

        self.conversation.append(
            {"role": "assistant", "content": bot_reply}
        )

        return bot_reply

    def classify_lead(self):
        classify_prompt = self.conversation + [
            {
                "role": "user",
                "content": """
Based on this conversation, classify the lead.

Return ONLY valid JSON in this format:
{
    "quality": "",
    "action": "",
    "reason": ""
}
"""
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=classify_prompt
        )

        return json.loads(response.choices[0].message.content)


# ---------------- MAIN ---------------- #

agent = LeadAgent()

print("Bot:", agent.respond("Hi I am interested in UPSC coaching"))
print("Bot:", agent.respond("I have been preparing for 2 years, serious about it"))
print("Bot:", agent.respond("Budget is not an issue, I want to start this month"))

print("\n--- FINAL CLASSIFICATION ---")

result = agent.classify_lead()

print("Quality:", result["quality"])
print("Action:", result["action"])
print("Reason:", result["reason"])
