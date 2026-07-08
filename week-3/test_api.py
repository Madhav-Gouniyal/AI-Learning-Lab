import requests

url = "https://ai-learning-lab-production.up.railway.app/ask"
data = {"question": "what are the fees"}

response = requests.post(url, json=data)
print(f"Status code: {response.status_code}")
print(f"Response: {response.text}")


