# app.py — Singapore Case Law RAG with Chainlit (beautiful & expandable)
import os, shutil, re
from glob import glob
from typing import List
from dotenv import load_dotenv

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
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Put your OpenRouter key in a .env file!")

llm = ChatOpenAI(
    model="x-ai/grok-4.1-fast:free",
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
if os.path.exists(CHROMA_PATH):
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

# =========================== LOAD & INDEX PDFs ===========================
def load_and_index():
    docs = []
    for pdf_path in glob(f"{PDF_FOLDER}/*.pdf"):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
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
    print(f"Indexed {len(docs)} pages")

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