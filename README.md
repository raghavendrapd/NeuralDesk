# NeuralDesk

Ask questions to any PDF and get answers back. That's it.

Under the hood it's a RAG pipeline — it breaks your document into chunks, stores them in a vector database, and when you ask something it finds the most relevant chunks and passes them to an LLM to answer. No hallucinations, no made-up facts — it only answers from what's actually in your document.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-green?style=flat-square&logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-purple?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-red?style=flat-square)

---

## Why I built this

I wanted to understand how RAG actually works — not just the theory but the full pipeline from PDF to answer. So I built it from scratch. ChromaDB for vector storage, SentenceTransformers for embeddings, Groq for inference, FastAPI to expose it as an API with a simple chat UI on top.

Works well for research papers, reports, contracts, resumes — anything where you need to extract specific information without reading the whole thing.

---

## How it works

```
PDF → split into chunks → embed each chunk → store in ChromaDB
                                                      │
                                        User asks a question
                                                      │
                                        Query gets embedded
                                                      │
                                   Top 8 similar chunks retrieved
                                                      │
                                    Groq LLM answers using chunks
```

The key detail is `k=8` — it retrieves the 8 most semantically similar chunks before answering. I found this matters a lot for aggregation questions like "how many X are there" where the answer is spread across different parts of the document.

---

## Stack

- **FastAPI** — backend + REST endpoints
- **ChromaDB** — vector store for embeddings
- **SentenceTransformers** — `all-MiniLM-L6-v2` for generating embeddings
- **Groq** — LLaMA 3.3 70B for generation (fast + free)
- **LangChain** — PDF loading and text splitting
- **Vanilla JS** — chat UI, nothing fancy

---

## Setup

```bash
git clone https://github.com/raghavendrapd/NeuralDesk.git
cd NeuralDesk

python -m venv venv
venv\Scripts\activate  # windows
source venv/bin/activate  # mac/linux

pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers fastapi uvicorn pypdf python-dotenv groq python-multipart
```

Add a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Free key at [console.groq.com](https://console.groq.com)

Drop a PDF into `/data`, then:
```bash
python ingest.py            # processes the PDF
uvicorn app:app --reload    # starts the server
```

Open https://neuraldesk.onrender.com and start asking questions.

---

## API

**POST `/ask`**
```json
{ "question": "What are the main conclusions?" }
```
```json
{ "answer": "..." }
```

**POST `/upload`**
Send a PDF as `multipart/form-data`. It gets processed and added to the existing vector store immediately.

---

## Project structure

```
NeuralDesk/
├── data/           # put PDFs here
├── vectorstore/    # chromadb storage, auto-created
├── ingest.py       # pdf → chunks → embeddings → chromadb
├── query.py        # cli interface for quick testing
├── app.py          # fastapi app + chat ui
└── .env            # api keys, not committed
```

---

## What I'd improve

- Streaming responses instead of waiting for the full answer
- Better chunking strategy for tables and structured data in PDFs
- Source attribution — show which page/chunk the answer came from
- Docker setup for easier deployment

---

## License

MIT
