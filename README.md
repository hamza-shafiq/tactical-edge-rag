# Tactical Edge RAG

This repository is responsible for `Chatbot Service` to answer user queries by leveraging the ingested data.

## Infrastructure
A typical RAG application has two main components:
- Indexing
- Retrieval and generation

## Major modules
- Developed the chatbot application to make use of the openAI model for the Q&A.
- Implemented the Langchain framework using the LLM model for thoughts, analysis & observation.
- Created our own embedding using Pinecone vector store to ingest a provided PDF document
- Implemented a Simple Chat Interface along with document upload functionality according to the requirements

## Project Setup

Clone the project repository & change the directory to go inside the codebase
```sh
git clone https://github.com/hamza-shafiq/tactical-edge-rag.git
cd tactical-edge-rag
```

Create & activate the virtual environment
```sh
python3 -m venv venv
source venv/bin/activate
```

Install required packages
```sh
pip install -r requirements.txt
```

Run application
```sh
python3 app.py
```
