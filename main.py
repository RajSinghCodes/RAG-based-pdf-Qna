from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

# -----------------------------
# STEP 0 — Gemini Setup
# -----------------------------
client = genai.Client(api_key="API_KEY")

# -----------------------------
# STEP 1 — Load PDF
# -----------------------------
file_path = "C:/Users/Admin/Desktop/ai path/pdf qna/ai_notes.pdf"

loader = PyPDFLoader(file_path)
documents = loader.load()

print("Documents loaded:", len(documents))

# -----------------------------
# STEP 2 — Chunking
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))

# -----------------------------
# STEP 3 — Embedding Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# STEP 4 — Chroma DB
# -----------------------------
client_db = chromadb.Client()
collection = client_db.create_collection(name="my_docs")

# -----------------------------
# STEP 5 — Store Embeddings
# -----------------------------
for i, chunk in enumerate(chunks):
    text = chunk.page_content
    vector = model.encode(text)

    collection.add(
        ids=[str(i)],
        embeddings=[vector.tolist()],
        documents=[text],
        metadatas=[chunk.metadata]
    )

print("Stored all chunks in vector DB")

#memory
chat_history = []

# -----------------------------
# STEP 6 — CHAT LOOP
# -----------------------------
print("\n🔹 Chat with your PDF (type 'exit' to quit)\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    # Convert query → embedding
    query_embedding = model.encode(query)

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=2
    )

    # Combine context
    context = " ".join(results["documents"][0])

    # Create prompt
    history_text = "\n".join(chat_history)

    prompt = f"""
You are an AI assistant.

Use the conversation history and context to give a better answer.

Conversation History:
{history_text}

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer using Gemini
    response = client.models.generate_content(
    model="gemini-3-flash-preview", 
    contents=prompt
    )
    answer = response.text

    print("\nAnswer:\n", answer)
    print("\n" + "-"*50 + "\n")
    # STORE MEMORY
    chat_history.append(f"User: {query}")
    chat_history.append(f"AI: {answer}")
