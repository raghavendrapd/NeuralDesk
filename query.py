"""
NeuralDesk - query.py
Takes a user question, finds the most relevant chunks
from ChromaDB, and answers using Groq LLM.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from groq import Groq

load_dotenv()

CHROMA_DB_PATH = "vectorstore"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_vectorstore():
    """Load the existing ChromaDB vectorstore."""
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore


def retrieve_context(vectorstore, question, k=8):
    """Find the top k most relevant chunks for the question."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


def ask_groq(question, context):
    """Send question + context to Groq and get an answer."""
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""You are NeuralDesk, an intelligent document assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Groq's fastest free model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,                    # low temp = more factual answers
        max_tokens=512,
    )

    return response.choices[0].message.content


def main():
    print("\n🧠 NeuralDesk — AI Document Assistant")
    print("=" * 45)
    print("Type your question or 'exit' to quit\n")

    vectorstore = load_vectorstore()

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("👋 Goodbye!")
            break

        print("\n⚙️  Searching document...")
        context = retrieve_context(vectorstore, question)

        print("🤖 NeuralDesk: ", end="")
        answer = ask_groq(question, context)
        print(answer)
        print()


if __name__ == "__main__":
    main()