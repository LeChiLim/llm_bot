# openrouter_router_example.py
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

# Your OpenRouter setup
llm = ChatOpenAI(
    model="x-ai/grok-4.1-fast:free",        # ← change to claude, gemini, etc. anytime
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# ── Two paths ─────────────────────────────
def say_hello(state):
    return {"messages": [{"role": "ai", "content": "Hello there! How can I help you today?"}]}

def say_goodbye(state):
    return {"messages": [{"role": "ai", "content": "Goodbye! Take care and come back soon"}]}

# ── Router: Now uses LLM (Grok via OpenRouter) to decide! ─────
def router(state):
    msg = state["messages"][-1].content

    prompt = f"""
Look at this user message and choose exactly one option:

Message: "{msg}"

Reply with ONLY one of these words, nothing else:
HELLO   → if user says hi, hello, morning, etc.
GOODBYE → if user says bye, goodbye, see you, etc.
ANYTHING ELSE → reply HELLO

Answer:"""

    decision = llm.invoke(prompt).content.strip().upper()
    
    # Fallback just in case
    if decision not in ["HELLO", "GOODBYE"]:
        decision = "HELLO"

    return {"next": decision}

# ── Build the graph ───────────────────────
graph = StateGraph(MessagesState)
graph.add_node("router", router)
graph.add_node("say_hello", say_hello)
graph.add_node("say_goodbye", say_goodbye)

# This is the magic line — AI chooses the path!
graph.add_conditional_edges(
    "router",
    lambda state: state["next"],           # ← LLM decides this value
    {
        "HELLO": "say_hello",
        "GOODBYE": "say_goodbye"
    }
)

graph.add_edge(START, "router")
graph.add_edge("say_hello", END)
graph.add_edge("say_goodbye", END)

app = graph.compile()

# ── Test it! ─────────────────────────────
if __name__ == "__main__":
    print("Say something →")
    user_input = input("> ")

    result = app.invoke({"messages": [{"role": "user", "content": user_input}]})
    print("\nAI says:")
    print(result["messages"][-1].content)