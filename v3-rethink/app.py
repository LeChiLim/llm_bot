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
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# =========================== CONFIG ===========================
PDF_FOLDER = "/home/asus_laptop/pdfs"          # ← change only if needed

# Set these flags to control re-indexing and summary/embedding regeneration
RESET_CHROMA_ON_START = True    # If True, deletes chroma_db and re-embeds all on start
REGENERATE_EMBEDDINGS = True    # If True, always recompute PDF summaries/embeddings

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Put your OpenRouter key in a .env file!")

llm = ChatOpenAI(
    model="amazon/nova-2-lite-v1:free",
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
docstore = InMemoryStore()

child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

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
- Cross-reference every relevant case.
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
    # Use the LLM to determine if it's a web search type query.
    routing_prompt = f"""
    Review the following user query and decide whether it needs:
    - Web search ('WEBSEARCH') if it's about general knowledge, definitions, or outside the PDF/context database.
    - RAG ('RAG') if it's likely to be case law/legal based and should use the PDF/context.
    - GENERIC ('GENERIC') for any other simple conversation.

    Query: {query}
    Reply with only: WEBSEARCH, RAG, or GENERIC
    """
    decision = llm.invoke(routing_prompt).content.strip().upper()
    if decision not in ["WEBSEARCH", "RAG", "GENERIC"]:
        decision = "RAG"  # Default to RAG for now if unknown
    return {"next": decision, "messages": state["messages"]}

def generic_state(state):
    # TODO: Add generic state logic here
    pass

def rag_state(state):
    query = state["messages"][-1]["content"].strip().lower()
    # Extract possible PDF/case mentions from the query
    mentioned = []
    for filename in pdf_summaries.keys():
        namepart = filename.replace(".pdf", "").lower()
        if namepart in query:
            mentioned.append(filename)
    context_blocks = []
    if mentioned:
        context_blocks = [pdf_summaries[f] for f in mentioned]
    else:
        # If nothing matched, fall back to all summaries
        context_blocks = list(pdf_summaries.values())
    # Compose a new prompt for the LLM
    rag_prompt = f"""
    Answer as a legal expert using only the following PDF summaries for context.

    Summaries:\n\n{chr(10).join(context_blocks)}\n\nQuestion: {query}\n\nAnswer:
    """
    answer = llm.invoke(rag_prompt).content.strip()
    return {"messages": state["messages"] + [{"role": "ai", "content": answer}]}

def websearch_state(state: Dict):
    query = state["messages"][-1]["content"].strip()
    # Placeholder LLM call for web search answer
    websearch_prompt = f"""
    You are an advanced assistant. The user has asked a question that likely requires up-to-date web search or general world knowledge beyond the provided case law PDFs.
    Please provide a direct, factual answer and cite any verifiable sources or say 'I cannot locate this information currently.'

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

# =========================== CHAINLIT UI ===========================
@cl.on_chat_start
async def start():
    await cl.Message(
        content="Singapore Case Law RAG ready ⚖️\nAsk me anything about the loaded judgments."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()

    # Special command
    if "list all cases" in query.lower():
        cases = sorted({d.metadata["citation"] for d in retriever.docstore.yield_keys() if "citation" in retriever.docstore.search(d.metadata["citation"])[0].metadata})
        await cl.Message(content="Available cases:\n\n" + "\n".join(f"• {c}" for c in cases)).send()
        return

    # Retrieve
    docs = retriever.invoke(query)

    # Right-side citation panel
    elements = [
        cl.Text(
            name=f"Source {i+1}",
            content=f"**{doc.metadata['citation']}** (p.{doc.metadata['page']})\n\n{doc.page_content[:1500]}{'...' if len(doc.page_content)>1500 else ''}",
            display="side"
        )
        for i, doc in enumerate(docs[:8])
    ]

    # Stream the answer
    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream({"question": query}):
        await msg.stream_token(chunk)

    await msg.update()

    # Attach the sources on the right after answer finishes
    await msg.update(elements=elements)