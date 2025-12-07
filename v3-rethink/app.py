# app.py — Singapore Case Law RAG with Chainlit (beautiful & expandable)
import os, shutil, re
from glob import glob
from typing import List, Dict
from dotenv import load_dotenv
from collections import defaultdict

import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import asyncio  # <-- already imported in app, ensure it's present

# =========================== CONFIG ===========================
PDF_FOLDER = "../PDFs"          # ← change only if needed

# Set these flags to control re-indexing and summary/embedding regeneration
RESET_CHROMA_ON_START = False    # If True, deletes chroma_db and re-embeds all on start
REGENERATE_EMBEDDINGS = False    # If True, always recompute PDF summaries/embeddings

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Put your OpenRouter key in a .env file!")

llm = ChatOpenAI(
    model="tngtech/deepseek-r1t-chimera:free",
    temperature=0.2,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=16000,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},          # ← set to "cuda" if you have GPU
)

CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")
if RESET_CHROMA_ON_START and os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Store PDF summaries
pdf_summaries = {}

def summarize_pdf(pdf_path, pages):
    """Use LLM to generate a summary for a PDF."""
    full_text = '\n'.join(page.page_content.strip() for page in pages)
    prompt = f"""
    Summarize the following legal judgment for future semantic search and context selection.
    Include:
      - Case name, court, and year
      - Most important holdings
      - Notable citations/references
      - Brief summary of facts
    
    Judgment text:
    {full_text[:10000]}  # Limit for LLM context window - can be tuned
    """
    summary = llm.invoke(prompt).content.strip()
    return summary

# =========================== LOAD & INDEX PDFs (with summary) ===========================
def load_and_index():
    docs = []
    if not REGENERATE_EMBEDDINGS and os.path.exists(CHROMA_PATH):
        print("Chroma DB exists and regeneration is off. Skipping embedding/summarization.")
        return  # Skip indexing, use what's there
    for pdf_path in glob(f"{PDF_FOLDER}/*.pdf"):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Generate summary with LLM
        summary = summarize_pdf(pdf_path, pages)
        pdf_summaries[os.path.basename(pdf_path)] = summary
        for i, page in enumerate(pages):
            if not page.page_content.strip():
                continue
            first_line = page.page_content.split("\n", 1)[0]
            citation = re.search(r'\[\d{4}\]\s*SG[A-Z]{2,4}\s*\d+', first_line)
            citation = citation.group(0).strip() if citation else os.path.basename(pdf_path)
            page.metadata.update({
                "source": os.path.basename(pdf_path),
                "page": i + 1,
                "citation": citation
            })
            docs.append(page)
    retriever.add_documents(docs)
    print(f"Indexed {len(docs)} pages with summaries")

load_and_index()

# =========================== RAG CHAIN ===========================
template = """You are a former Judicial Commissioner of the Supreme Court of Singapore with 30+ years experience.

Instructions (strict):
- Extremely detailed, nuanced, fully reasoned answers only.
- Be brief and concise but also provide a comprehensive answer.
- Use full neutral citations: [2023] SGCA 12
- Quote exact paragraphs/pages.
- Highlight overrulings/developments.
- End with **Summary Holding** in bold.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
def format_docs(docs):
    # (no change)
    return "\n\n".join(
        f"[{d.metadata['citation']}] (p.{d.metadata['page']})\n{d.page_content.strip()}"
        for d in docs
    )

retriever_with_format = retriever | format_docs # (no change)

chain = (
    {
        # FIX: Put the lambda in parentheses to ensure it's treated as a runnable before piping
        "context": (lambda x: x["question"]) | retriever_with_format, 
        # The question lambda is just for extracting the variable for the prompt
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =========================== LLM ROUTER (skeleton) ===========================
from langgraph.graph import StateGraph, MessagesState, START, END

# Define node functions as placeholders

def llm_router(state: Dict):
    query = state["messages"][-1]["content"].strip()
    # Identify if the query is about a specific case
    # If any case/citation/pdf matches, RAG, else if legal/general topics, Generic, else WebSearch.
    lower_query = query.lower()
    matched = False
    for filename in pdf_summaries.keys():
        namepart = filename.replace(".pdf", "").lower()
        if namepart in lower_query:
            matched = True
            break
    routing_prompt = f"""
    Classify the user question:
    - Reply with 'RAG' if the query asks about or references specific court cases, judgments, pdfs, or citations.
    - Reply with 'GENERIC' if the user asks about legal topics, types of cases, common patterns, or summaries (not mentioning a specific case).
    - Reply 'WEBSEARCH' if the question is clearly not related to anything in the PDFs and is world/general knowledge.
    If you are unsure, reply with 'WEBSEARCH'.
    Query: {query}
    """
    # Prefer our fast logic, but let LLM correct if ambiguous.
    if matched:
        decision = "RAG"
    else:
        decision = llm.invoke(routing_prompt).content.strip().upper()
        if decision not in ["WEBSEARCH", "RAG", "GENERIC"]:
            decision = "GENERIC"
    return {"next": decision, "messages": state["messages"]}

def generic_state(state):
    # Send all summaries to LLM, let it pick context
    query = state["messages"][-1]["content"].strip()
    context_blocks = list(pdf_summaries.values())
    generic_prompt = f"""
    The user asked a general legal question. Using ONLY these case summaries, answer as an expert:
    Be brief and concise but also provide a comprehensive answer!
    Summaries:\n\n{chr(10).join(context_blocks)}\n\nQuestion: {query}\n\nAnswer:
    """
    answer = llm.invoke(generic_prompt).content.strip()
    return {"messages": state["messages"] + [{"role": "ai", "content": answer}]}

def rag_state(state):
    # If mentions cases/pdfs/citations, run vector search. Else fallback to all summaries.
    query = state["messages"][-1]["content"].strip()
    lower_query = query.lower()
    mentioned = []
    for filename in pdf_summaries.keys():
        namepart = filename.replace(".pdf", "").lower()
        if namepart in lower_query:
            mentioned.append(filename)
    if mentioned:
        # Similarity search on subset of docs (those matching cases)
        docs = []
        for file in mentioned:
            sub_docs = [d for d in vectorstore.yield_keys() if file == d.metadata.get("source")]
            docs.extend(sub_docs)
        if not docs:
            docs = retriever.invoke(query)  # fallback
    else:
        docs = retriever.invoke(query)  # Use semantic search for query
    context = "\n\n".join(
        f"[{d.metadata['citation']}] (p.{d.metadata['page']})\n{d.page_content.strip()}" for d in docs[:6]
    )
    rag_prompt = f"""
    Act as a Singapore legal expert. Answer the question using ONLY the following extracts and citations. Do not speculate:
    Be brief and concise but also provide a comprehensive answer!
    
    Sources:\n\n{context}\n\nQuestion: {query}\n\nAnswer:
    """
    answer = llm.invoke(rag_prompt).content.strip()
    return {"messages": state["messages"] + [{"role": "ai", "content": answer}]}

def websearch_state(state: Dict):
    query = state["messages"][-1]["content"].strip()
    # Placeholder LLM call for web search answer
    websearch_prompt = f"""
    You are an advanced assistant. The user has asked a question that likely requires up-to-date web search or general world knowledge beyond the provided case law PDFs.
    Please provide a direct, factual answer and cite any verifiable sources or say 'I cannot locate this information currently.'

    Be brief and concise but also provide a comprehensive answer!
    Question: {query}

    Answer:
    """
    answer = llm.invoke(websearch_prompt).content.strip()
    return {"messages": state["messages"] + [{"role": "ai", "content": answer}]}

# Build the graph (structure only)
graph = StateGraph(MessagesState)
graph.add_node("llm_router", llm_router)
graph.add_node("generic_state", generic_state)
graph.add_node("rag_state", rag_state)
graph.add_node("websearch_state", websearch_state)

graph.add_conditional_edges(
    "llm_router",
    lambda state: state.get("next"),
    {
        "GENERIC": "generic_state",
        "RAG": "rag_state",
        "WEBSEARCH": "websearch_state"
    }
)

graph.add_edge(START, "llm_router")
graph.add_edge("generic_state", END)
graph.add_edge("rag_state", END)
graph.add_edge("websearch_state", END)

# Compile graph (skeleton)
lm_router_app = graph.compile()

# ========= Add loader animation util =========

async def send_animated_message(
    base_msg: str,
    frames: list,
    interval: float = 0.8
) -> 'cl.Message':
    msg = cl.Message(content=base_msg)
    await msg.send()
    progress = 0
    bar_length = 12
    try:
        while True:
            current_frame = frames[progress % len(frames)]
            progress_bar = ("▣" * (progress % bar_length)).ljust(bar_length, "▢")
            new_content = f"{current_frame} {base_msg}\n{progress_bar}"
            msg.content = new_content
            await msg.update()
            progress += 1
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # Final static message (blank or fallback)
        msg.content = base_msg
        await msg.update()
    return msg

# =========================== CHAINLIT UI ===========================
import time

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Singapore Case Law RAG ready ⚖️\nAsk me anything about the loaded judgments."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    start_time = time.time()

    # Use Chainlit Step with glowing loader, just like DeepSeek example
    async with cl.Step(name="Brendan's Big Brain to think...") as step:
        async for chunk in chain.astream({"question": query}):
            await step.stream_token(chunk)
        await step.update()  # Finalize the glowing step with all LLM tokens

    elapsed = time.time() - start_time

    # Show sources in a sidebar as before
    docs = retriever.invoke(query)
    elements = [
        cl.Text(
            name=f"Source {i+1}",
            content=f"**{doc.metadata['citation']}** (p.{doc.metadata['page']})\n\n{doc.page_content[:1500]}{'...' if len(doc.page_content)>1500 else ''}",
            display="side"
        )
        for i, doc in enumerate(docs[:8])
    ]
    await cl.Message(
        content=f"---\n<sub>Query completed in {elapsed:.2f} seconds.</sub>",
        elements=elements,
        author="AI"
    ).send()