# === 1. Imports ===
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama   # Option A: Ollama (fast & local)
# from langchain_huggingface import HuggingFacePipeline  # Option B: local HF model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import shutil
from glob import glob

# === 2. Settings (change these) ===
PDF_FOLDER = "/home/asus_laptop/pdfs"          # Put your PDFs here
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model (free, works offline after first download)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM choice
# Option A - Ollama (recommended - fast, free, fully local)
print("Lets initiate the LLM!")
llm = ChatOllama(
    model="llama3.2:3b",   # This exists after you pulled it
    temperature=0.2,
    base_url="http://localhost:11434"  # explicit is better
)

print("LLM created successfully!")

# Simple test
response = llm.invoke("Say: 'Hello! Ollama is working!' in French.")
print("\nResponse:")
print(response.content)


# Option B - HuggingFace local (uncomment if you prefer)
# from langchain_huggingface import HuggingFacePipeline
# from transformers import pipeline
# hf_pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", max_new_tokens=1024, device=0)
# llm = HuggingFacePipeline(pipeline=hf_pipe)

# === 3. Load & split PDFs ===
def load_pdfs():
    docs = []
    for pdf_path in glob(os.path.join(PDF_FOLDER, "*.pdf")):
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    return docs


print("Loading PDFs...")
documents = load_pdfs()
print(f"Loaded {len(documents)} pages from {len(glob(os.path.join(PDF_FOLDER, '*.pdf')))} PDFs")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
splits = text_splitter.split_documents(documents)
print(f"Created {len(splits)} chunks")

# === 4. Create vector store (only once, then persists) ===
# Check if we have documents to index
if len(splits) == 0:
    print("ERROR: No documents found! Please add PDF files to the PDF_FOLDER.")
    print(f"Looking for PDFs in: {PDF_FOLDER}")
    print("Exiting...")
    exit(1)

# Remove corrupted chroma_db if it exists to avoid KeyError
chroma_db_path = "./chroma_db"
if os.path.exists(chroma_db_path):
    try:
        # Try to load existing vectorstore to check if it's valid
        test_vectorstore = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embedding_model
        )
        # If we get here, the database is valid, use it
        print("Using existing vectorstore...")
        vectorstore = test_vectorstore
    except (KeyError, Exception) as e:
        print(f"Existing chroma_db appears corrupted ({type(e).__name__}: {str(e)}), removing it...")
        shutil.rmtree(chroma_db_path)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=chroma_db_path
        )
        print("Created new vectorstore.")
else:
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=chroma_db_path   # saves to disk so you don't re-index every time
    )
    print("Created new vectorstore.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# === 5. RAG prompt ===
template = """You are a helpful assistant. Answer the question using only the following context.
If you cannot answer based on the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# === 6. RAG chain ===
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# === 7. Gradio Chat UI ===
def chat(message, history):
    # Stream the response
    full_response = ""
    for chunk in rag_chain.stream(message):
        full_response += chunk
        yield full_response

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="PDF RAG Chatbot 🤖",
    description=f"Currently indexed {len(glob(os.path.join(PDF_FOLDER, '*.pdf')))} PDFs with {len(splits)} chunks.",
    examples=["What is the main topic of the documents?", "Summarize the first PDF", "Who is the author?"],
    cache_examples=False,
)

# === 8. Launch ===
demo.launch(share=True)   # share=True gives you a public link (great for testing)