"""
NeuralDesk - ingest.py
Loads PDFs from the /data folder, splits them into chunks,
and stores them in a ChromaDB vector database.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

DATA_FOLDER = "data"
CHROMA_DB_PATH = "vectorstore"


def load_pdfs(folder):
    """Load all PDFs from the data folder."""
    documents = []
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDFs found in /data folder. Drop a PDF in there first!")
        return []

    for pdf_file in pdf_files:
        path = os.path.join(folder, pdf_file)
        print(f"📄 Loading: {pdf_file}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    print(f"✅ Loaded {len(documents)} pages from {len(pdf_files)} PDF(s)")
    return documents


def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # each chunk = 500 characters
        chunk_overlap=50,      # 50 char overlap so we dont lose context
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


def store_in_vectordb(chunks):
    """Convert chunks to embeddings and store in ChromaDB."""
    print("⚙️  Creating embeddings and storing in ChromaDB...")

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"   # fast, lightweight, free embedding model
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"✅ Stored in ChromaDB at ./{CHROMA_DB_PATH}")
    return vectorstore


def main():
    print("\n🚀 NeuralDesk — Document Ingestion Pipeline")
    print("=" * 45)

    documents = load_pdfs(DATA_FOLDER)
    if not documents:
        return

    chunks = split_documents(documents)
    store_in_vectordb(chunks)

    print("\n✅ Ingestion complete! You can now run query.py to ask questions.")


if __name__ == "__main__":
    main()