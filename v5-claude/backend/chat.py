import sys
import json
from dotenv import load_dotenv
import requests
import os

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "liquid/lfm-2.5-1.2b-instruct:free"         # cheap + fast for testing

def chat(messages: list) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Harvey Lite, a legal AI assistant specialising in Hong Kong and Singapore case law. Be concise and cite cases where relevant."
                },
                *messages
            ],
            "stream": True,
        },
        stream=True
    )

    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    # Print each chunk as SSE for Next.js to forward
                    print(f"data: {json.dumps({'type': 'text', 'content': content})}", flush=True)
            except:
                continue

if __name__ == "__main__":
    # Read messages from stdin (sent by Next.js)
    #messages = json.loads(sys.stdin.read())
    messages = [
        {"role": "user", "content": "What are the elements of negligence in Hong Kong law?"}
    ]
    chat(messages)