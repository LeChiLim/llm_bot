# llm_bot
This is to learn about RAG pipelines and LLM. 

The project contains a RAG pipeline to read from a set number of PDFs, using
LangChan, ChromaDB, and OpenRouter. 

python3 - m venv venv

source venv/bin/activate

pip install -r requirements first!

Then just python3 test.py or python3 test1.py accordingly.

v1-local contains a RAG pipeline that runs locally; initiating a LLM using Ollama.
You may need to pip install ollama and run it seperately. 

curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
ollama serve

v1-online uses OpenRouter to connect to Grok v4.1 (which is free). 
Please create a .env file in the folder of the python file. Then get the API keys from openrouter

-----
v2 onwards uses uv as a package manager. Please find the requirements.in file inside each version folder.

To install uv;
curl -LsSf https://astral.sh/uv/install.sh | sh

#create venv
uv venv 
uv pip sync requirements-lock.txt

v2-chainlit uses chainlit as the front end simimlar to v1-online.

run: chainlit run pipeline.py -w --host 0.0.0.0
