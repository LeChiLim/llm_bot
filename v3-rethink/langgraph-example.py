from langgraph.graph import StateGraph, MessagesState, START, END

def greeter(state):
    return {"messages": [{"role": "ai", "content": "Hi! I'm Grok."}]}

def answerer(state):
    return {"messages": [{"role": "ai", "content": "The answer is 42."}]}

graph = StateGraph(MessagesState)
graph.add_node("greeter", greeter)
graph.add_node("answerer", answerer)

graph.add_edge(START, "greeter")
graph.add_edge("greeter", "answerer")   # new syntax
graph.add_edge("answerer", END)

app = graph.compile()

result = app.invoke({"messages": [{"role": "user", "content": "hello"}]})
print(result["messages"])
