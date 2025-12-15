from langchain_ollama import ChatOllama
import datetime

print("Let's initiate the LLM!")

# Record start time
start_time = datetime.datetime.now()

llm = ChatOllama(
    model="qwen3:4b",   # This exists after you pulled it
    temperature=0.2,
    base_url="http://100.127.11.87:11434"  # explicit is better
)

print("LLM created successfully!")

response = llm.invoke("Tell me a funny story. Make it 300 words long.")

# Record end time and calculate duration
end_time = datetime.datetime.now()
duration = end_time - start_time

print("\nResponse:")
print(response.content)
print(f"\nQuery took: {duration.total_seconds():.2f} seconds")
