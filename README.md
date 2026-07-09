# Production RAG Application (OpenRouter + LangChain)

A production-grade Retrieval-Augmented Generation (RAG) system built with LangChain, OpenRouter, and ChromaDB.

## Features

- **Hybrid Retrieval**: Combines sparse keyword search (BM25) and dense vector search (embeddings) to capture both semantic meaning and exact matches.
- **Cross-Encoder Reranking**: Uses `BAAI/bge-reranker-base` to rerank retrieved document chunks, maximizing context relevance.
- **Citation-Enforced Answering**: Restricts generation strictly to the provided context and enforces inline source citations (e.g., `[S1]`).
- **Observability**: Built-in tracing with local JSONL file logging and Langfuse / LangSmith integrations.
- **Metrics Dashboard**: Tracks per-request latency, retrieval/generation execution times, token counts, and API costs.
- **Evaluation Regression Gates**: Automated quality checks validating semantic faithfulness, answer relevance, context precision, and citation coverage against standard goldens datasets.
- **FastAPI Backend & Streamlit Frontend**: Access the application through a web API or a rich chat-based dashboard.

---

## Project Structure

```text
production_rag_openrouter_project/
├─ app.py                # Streamlit user interface
├─ main.py               # FastAPI application backend server
├─ document_loader.py    # PDF loading and metadata extraction
├─ vector_store.py       # Chroma & BM25 indexing/search
├─ rag_chain.py          # RAG response generation and citation validation
├─ chat_store.py         # Local JSON-based chat session manager
├─ llm.py                # LangChain OpenRouter client initialization
├─ config.py             # Settings, threshold parameters, and settings schema
├─ observability.py      # Tracing callbacks for local logging, Langfuse, and LangSmith
├─ metrics.py            # Latency, costs, and tokens tracking store
├─ costs.py              # Cost computation utilities
├─ requirements.txt      # Python dependencies
├─ .env.example          # Sample environment configuration template
├─ .gitignore            # Git exclusion rules
├─ evals/                # Quality evaluation suite
│  ├─ run_eval.py        # Semantic evaluation runner and gate
│  └─ goldens.jsonl      # Baseline ground truth evaluation dataset
├─ tests/                # Unit test suite
│  ├─ test_citations.py  # Tests citation logic
│  └─ test_sources.py    # Tests tokenization & deduplication
└─ .github/
   └─ workflows/
      └─ ci.yml          # GitHub Actions CI pipeline
```

---

## Setup Instructions

### 1. Clone & Set Up Environment
```bash
# Clone your repository (once uploaded)
git clone <your-repo-url>
cd production_rag_openrouter_project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (CMD):
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your OpenRouter API key and other configurations:
```bash
cp .env.example .env
```
Ensure `OPENROUTER_API_KEY` is set in `.env` along with any desired observability keys (Langfuse or LangSmith).

---

## Running the Application

This project provides two ways to run and interact with the RAG system:

### 1. Streamlit Dashboard (Frontend UI)
Run the interactive chatbot UI:
```bash
streamlit run app.py
```
This launches a browser window (typically at `http://localhost:8501`) where you can:
- Upload and index PDF documents.
- Start and switch between multiple chat history sessions.
- Ask questions and view answers, inline citations, source snippet metadata, latencies, and request costs.

### 2. FastAPI Web Server (Backend API)
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
Once started, the backend runs at `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to access interactive Swagger documentation.

#### Key Endpoints:
- **`GET /health`**: Health status check.
- **`POST /load-path`**: Index local files/folders.
  - Payload: `{"path": "data/document.pdf"}`
- **`POST /upload`**: Upload and index a PDF file dynamically via multipart form upload.
- **`POST /ask`**: Query the indexed documents.
  - Payload: `{"question": "What is the training set size?", "user_id": "user_1", "session_id": "session_1"}`
- **`GET /metrics`**: Retrieve logged performance metrics, token statistics, and cost totals.

---

## Quality Evaluation & CI Regression Gate

Run the unit tests and quality validation script before committing code:

```bash
# Run unit tests
python -m pytest -q

# Run RAG semantic evaluation suite
python -m evals.run_eval
```

The evaluation script generates metrics comparing RAG responses to a set of goldens (`evals/goldens.jsonl`). It enforces threshold values specified in `.env` or `config.py` for:
- Faithfulness
- Answer Relevance
- Context Precision
- Citation Coverage

If any metric falls below the threshold, the script exits with an error status (used as a quality regression gate in the CI workflow).
