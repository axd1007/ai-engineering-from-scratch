
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"Key loaded: {'yes' if api_key else 'no'}")
print(f"Key starts with: {api_key[:10]}..." if api_key else "Key not found")

client = anthropic.Anthropic()
print("Client created successfully")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
)

print(response.content[0].text)
