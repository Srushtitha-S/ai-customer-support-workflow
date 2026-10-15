
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load credentials from .env
load_dotenv(override=True)

token = os.getenv("HF_TOKEN")
model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

if not token:
    raise RuntimeError(
        "HF_TOKEN is missing. Check your .env file."
    )

client = InferenceClient(
    model=model,
    api_key=token.strip(),
    timeout=60,
)

response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": (
                "Classify this support ticket in one sentence: "
                "I was charged twice for my order."
            ),
        }
    ],
    max_tokens=100,
)

print("Hugging Face connection successful!")
print("\nModel response:")
print(response.choices[0].message.content)
