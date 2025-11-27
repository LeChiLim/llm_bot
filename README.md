# llm_bot
This is to learn about RAG pipelines and LLM. 

The project contains a RAG pipeline to read from a set number of PDFs, using
LangChan, ChromaDB, and OpenRouter. 


pip install -r requirements first!

Then just python3 test.py or python3 test1.py accordingly.

v1-local contains a RAG pipeline that runs locally; initiating a LLM using Ollama.
You may need to pip install Ollama and run it seperately. 

v1-online uses OpenRouter to connect to Grok v4.1 (which is free).
