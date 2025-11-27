import gradio as gr
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
from glob import glob
import os
import shutil
import re

# ===========================
# CONFIG – DEEPSEEK R1 FREE VIA OPENROUTER
# ===========================
PDF_FOLDER = "/home/asus_laptop/pdfs"

# Get your free key from https://openrouter.ai/keys
OPENROUTER_API_KEY = "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"  # ← PUT YOUR KEY HERE

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatOpenAI(
    model="x-ai/grok-4.1-fast:free",          # 100% free & very strong
    temperature=0.2,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=8000,
    default_headers={
        "HTTP-Referer": "http://localhost",   # optional but nice for leaderboard
        "X-Title": "Singapore Case Law RAG",
    }
)
print("DeepSeek R1 (free) connected!")

# ===========================
# Chroma DB
# ===========================
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
os.makedirs(CHROMA_PATH, exist_ok=True)

# ===========================
# Chunking
# ===========================
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# ===========================
# Vectorstore + Retriever
# ===========================
vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# ===========================
# Load PDFs – SAFELY (skip empty or None pages)
# ===========================
def load_docs():
    docs = []
    pdf_files = glob(f"{PDF_FOLDER}/*.pdf")
    print(f"Found {len(pdf_files)} PDFs")
    
    for pdf in pdf_files:
        print(f"Loading {os.path.basename(pdf)}")
        loader = PyPDFLoader(pdf)
        pages = loader.load()  # this returns list of Document objects
        
        for i, page in enumerate(pages):
            # ---- FIX: skip completely empty pages ----
            if not page.page_content or page.page_content.strip() == "":
                print(f"  → Skipping empty page {i+1}")
                continue
                
            # Extract Singapore citation from first line (robust)
            first_line = page.page_content.split("\n", 1)[0]
            match = re.search(r'\[\d{4}\]\s*SG[A-Z]{2,4}\s*\d+', first_line)
            citation = match.group(0).strip() if match else f"Page {i+1} of {os.path.basename(pdf)}"
            
            page.metadata.update({
                "source": os.path.basename(pdf),
                "page": i + 1,
                "citation": citation
            })
            docs.append(page)
    
    print(f"Successfully loaded {len(docs)} non-empty pages")
    return docs

docs = load_docs()

# ===========================
# Index
# ===========================
print("Indexing documents (30–90 seconds)...")
retriever.add_documents(docs)
print("Indexing complete!")

# ===========================
# RAG Chain
# ===========================
template = """You are an expert Singapore lawyer. Answer using ONLY the context below.
When asked to list cases, give full citations in format [YYYY] SGXX NNN.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata['citation']}] (Page {d.metadata['page']} | {d.metadata['source']})\n{d.page_content.strip()}"
        for d in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ===========================
# Gradio UI
# ===========================
def chat(message, history):
    result = ""
    for chunk in chain.stream(message):
        result += chunk
        yield result

print("Launching Gradio interface...")

gr.ChatInterface(
    fn=chat,
    title="Singapore Case Law RAG – DeepSeek R1 (100% FREE)",
    description="ParentDocumentRetriever + Chroma + DeepSeek R1 via OpenRouter – No cost, no crashes",
    examples=[
        "List all cases with full citations",
        "What is the test for contempt of court?",
        "Summarize [2018] SGHCR 9",
        "Compare the different types of contempt across the cases"
    ],
    theme="soft",
    retry_btn=None,           # removes the annoying retry button on errors
    undo_btn=None,
).launch(share=True)