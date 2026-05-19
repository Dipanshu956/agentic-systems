import os
import re
import uuid
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

# initialize local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================
# CONFIG
# ==========================
PROJECT_DIR = Path(__file__).parent
PDF_FOLDER = PROJECT_DIR / "policy_documents"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "campus_policies"




# ==========================
# POLICY TYPE
# ==========================
def infer_policy_type(filename):
    name = filename.lower()

    if "hostel" in name:
        return "hostel"
    elif "refund" in name:
        return "refund"
    elif "library" in name:
        return "library"
    else:
        return "general"


# ==========================
# CLEAN TEXT
# ==========================
def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================
# LOAD PDFs
# ==========================
def load_all_pdfs(folder):
    documents = []

    for pdf_file in folder.glob("*.pdf"):
        reader = PdfReader(pdf_file)

        print(f"Loaded {len(reader.pages)} pages from: {pdf_file.name}")

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                documents.append({
                    "text": clean_text(text),
                    "source": pdf_file.name,
                    "page": page_num,
                    "policy_type": infer_policy_type(pdf_file.name)
                })

    return documents


# ==========================
# CHUNKING
# ==========================
def chunk_text(text, chunk_size=150, overlap=20):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks


# ==========================
# EMBEDDINGS
# ==========================
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# ==========================
# BUILD VECTOR DB
# ==========================
def build_knowledge_base():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = chroma_client.create_collection(COLLECTION_NAME)

    docs = load_all_pdfs(PDF_FOLDER)

    total_chunks = 0

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            emb = get_embedding(chunk)

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{
                    "source": doc["source"],
                    "page": doc["page"],
                    "policy_type": doc["policy_type"]
                }]
            )

            total_chunks += 1

    print(f"Total chunks created: {total_chunks}")
    print(f"Successfully stored {total_chunks} chunks in vector database.")

    return collection


# ==========================
# RETRIEVAL
# ==========================
def retrieve_chunks(collection, query, top_k=3):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results["documents"][0]

    print(f"Retrieved {len(docs)} relevant chunks.")

    return docs


# ==========================
# PROMPT BUILDER
# ==========================
def build_prompt(query, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""
You are a campus policy assistant.

Answer ONLY from the provided policy context.

If answer is missing, say:
"I don't have that information."

Keep answers simple and student-friendly.

POLICY CONTEXT:
{context}

STUDENT QUESTION:
{query}
"""
    return prompt


# ==========================
# FINAL QA
# ==========================
def answer_question(collection, query):
    chunks = retrieve_chunks(collection, query)

    prompt = build_prompt(query, chunks)

    result = generator(
        prompt,
        max_length=200,
        do_sample=False
    )

    return result[0]["generated_text"]


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    print(f"Vector DB ready. Collection: {COLLECTION_NAME}")

    collection = build_knowledge_base()

    test_queries = [
        "Can I get a refund after dropping a course?",
        "What is the deadline for returning a library book?",
        "Are hostel visitors allowed on weekends?"
    ]

    for q in test_queries:
        print("\n" + "="*60)
        print("User Query:", q)

        answer = answer_question(collection, q)

        print("Answer:", answer)