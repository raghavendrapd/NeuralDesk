"""
NeuralDesk - app.py
FastAPI web interface for the RAG pipeline.
Exposes a /ask endpoint and serves a clean chat UI.
"""

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import shutil
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

app = FastAPI(title="NeuralDesk", description="AI-powered document intelligence system")

CHROMA_DB_PATH = "vectorstore"
DATA_FOLDER = "data"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


class Question(BaseModel):
    question: str


def get_vectorstore():
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)


def ask_groq(question: str, context: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""You are NeuralDesk, an intelligent document assistant.
Use ONLY the context below to answer the question accurately.
If the answer is not found, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
#  API ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the chat UI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NeuralDesk</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f0f13; color: #e2e2e8; min-height: 100vh; display: flex; flex-direction: column; }
    header { padding: 18px 32px; background: #16161d; border-bottom: 1px solid #2a2a35;
             display: flex; align-items: center; gap: 12px; }
    .logo { width: 32px; height: 32px; background: linear-gradient(135deg, #6c63ff, #3ecfcf);
            border-radius: 8px; display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 14px; color: #fff; }
    header h1 { font-size: 18px; font-weight: 600; color: #fff; }
    header span { font-size: 12px; color: #6c6c80; margin-left: 4px; }
    .container { flex: 1; max-width: 780px; width: 100%; margin: 0 auto;
                 padding: 24px 16px; display: flex; flex-direction: column; gap: 16px; }
    .upload-box { background: #16161d; border: 1px dashed #2a2a35; border-radius: 12px;
                  padding: 20px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
    .upload-box label { font-size: 13px; color: #9090a0; }
    .upload-box input[type=file] { font-size: 13px; color: #c2c2d0; }
    .upload-btn { padding: 8px 18px; background: #6c63ff; color: #fff; border: none;
                  border-radius: 8px; font-size: 13px; cursor: pointer; transition: background .2s; }
    .upload-btn:hover { background: #5a52e0; }
    .upload-status { font-size: 12px; color: #3ecfcf; margin-top: 6px; width: 100%; }
    #chat-box { flex: 1; background: #16161d; border: 1px solid #2a2a35; border-radius: 12px;
                padding: 20px; display: flex; flex-direction: column; gap: 14px;
                min-height: 380px; max-height: 480px; overflow-y: auto; }
    .msg { display: flex; flex-direction: column; gap: 4px; max-width: 85%; }
    .msg.user { align-self: flex-end; align-items: flex-end; }
    .msg.bot  { align-self: flex-start; }
    .bubble { padding: 11px 16px; border-radius: 12px; font-size: 14px; line-height: 1.6; }
    .msg.user .bubble { background: #6c63ff; color: #fff; border-bottom-right-radius: 4px; }
    .msg.bot  .bubble { background: #1e1e2a; color: #e2e2e8; border-bottom-left-radius: 4px; border: 1px solid #2a2a35; }
    .msg-label { font-size: 11px; color: #555568; }
    .input-row { display: flex; gap: 10px; }
    #question { flex: 1; padding: 12px 16px; background: #16161d; border: 1px solid #2a2a35;
                border-radius: 10px; color: #e2e2e8; font-size: 14px; outline: none; }
    #question:focus { border-color: #6c63ff; }
    #ask-btn { padding: 12px 22px; background: #6c63ff; color: #fff; border: none;
               border-radius: 10px; font-size: 14px; cursor: pointer; transition: background .2s; }
    #ask-btn:hover { background: #5a52e0; }
    #ask-btn:disabled { background: #3a3a50; cursor: not-allowed; }
    .thinking { color: #6c6c80; font-style: italic; font-size: 13px; }
  </style>
</head>
<body>
<header>
  <div class="logo">N</div>
  <h1>NeuralDesk <span>— AI Document Intelligence</span></h1>
</header>
<div class="container">

  <!-- Upload Section -->
  <div class="upload-box">
    <label>📄 Upload a new PDF:</label>
    <input type="file" id="pdf-input" accept=".pdf"/>
    <button class="upload-btn" onclick="uploadPDF()">Upload & Process</button>
    <div class="upload-status" id="upload-status"></div>
  </div>

  <!-- Chat Section -->
  <div id="chat-box">
    <div class="msg bot">
      <span class="msg-label">NeuralDesk</span>
      <div class="bubble">👋 Hello! I'm NeuralDesk. Upload a PDF or ask me anything about your already loaded document.</div>
    </div>
  </div>

  <!-- Input -->
  <div class="input-row">
    <input type="text" id="question" placeholder="Ask anything about your document..." onkeydown="if(event.key==='Enter') askQuestion()"/>
    <button id="ask-btn" onclick="askQuestion()">Ask</button>
  </div>

</div>
<script>
  async function askQuestion() {
    const input = document.getElementById('question');
    const q = input.value.trim();
    if (!q) return;
    input.value = '';

    addMessage('user', q);
    const thinking = addMessage('bot', '⚙️ Searching document...', true);
    document.getElementById('ask-btn').disabled = true;

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });
      const data = await res.json();
      thinking.remove();
      addMessage('bot', data.answer);
    } catch(e) {
      thinking.remove();
      addMessage('bot', '❌ Error connecting to server.');
    }
    document.getElementById('ask-btn').disabled = false;
  }

  async function uploadPDF() {
    const fileInput = document.getElementById('pdf-input');
    const status = document.getElementById('upload-status');
    if (!fileInput.files.length) { status.textContent = '⚠️ Please select a PDF first.'; return; }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    status.textContent = '⏳ Processing PDF...';

    try {
      const res = await fetch('/upload', { method: 'POST', body: formData });
      const data = await res.json();
      status.textContent = '✅ ' + data.message;
    } catch(e) {
      status.textContent = '❌ Upload failed.';
    }
  }

  function addMessage(role, text, isTemp=false) {
    const box = document.getElementById('chat-box');
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `<span class="msg-label">${role === 'user' ? 'You' : 'NeuralDesk'}</span>
                     <div class="bubble ${isTemp ? 'thinking' : ''}">${text}</div>`;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
  }
</script>
</body>
</html>
"""


@app.post("/ask")
def ask(body: Question):
    """Accept a question and return an AI answer from the document."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(body.question)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = ask_groq(body.question, context)
    return {"answer": answer}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Upload a new PDF, process it and add to vectorstore."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    file_path = os.path.join(DATA_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    vectorstore.add_documents(chunks)

    return {"message": f"'{file.filename}' processed — {len(chunks)} chunks added!"}