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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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
template = """You are a highly experienced Singapore litigation lawyer with more than 25 years of practice, 
formerly a Judicial Commissioner of the Supreme Court of Singapore and now a door tenant at one of the top sets in Maxwell Chambers.

Instructions (follow exactly):
1. Always give extremely detailed, nuanced and comprehensive answers – never give short answers.
2. Actively cross-reference facts, ratios and principles across ALL the retrieved cases/PDFs. 
   If the same issue appears in multiple judgments, compare and contrast them explicitly.
3. When citing, always use the full neutral citation format, e.g. [2023] SGCA 12 or [2018] SGHCR 9.
4. Quote the exact paragraph or page when a key principle is stated.
5. If there is any development or overruling of earlier authority in a later case, highlight it clearly.
6. Use Singapore legal terminology precisely (e.g. “Mareva injunction”, “Anton Piller order”, “strike out under O 18 r 19”, etc.).
7. Structure longer answers with clear headings and numbered paragraphs when helpful.
8. If the question involves statutory interpretation, refer to s 9A of the Interpretation Act and Purposive Approach cases if they appear in the corpus.
9. End every answer with a short “Summary Holding” in bold.

Use ONLY the context below. If you cannot answer fully from the provided context, say clearly what is missing.

Context:
{context}

Question: {question}

Answer (comprehensive, cross-referenced, and fully reasoned):"""

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