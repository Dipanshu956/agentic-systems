# hr_policy_rag.py

import os
import chromadb
from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

# -----------------------------
# STEP 1: DEFINE HR DOCUMENTS
# -----------------------------
HR_POLICY_DOCUMENTS = [
    {
        "id": "leave_001",
        "text": """
        Employees are entitled to 20 annual leave days per calendar year.
        Sick leave of up to 10 days is available annually.
        A maximum of 5 unused leave days may be carried forward into the next year.
        Leave requests must be submitted through the HR portal.
        """,
        "metadata": {
            "category": "Leave Policy",
            "source": "HR Handbook"
        }
    },
    {
        "id": "wfh_001",
        "text": """
        Employees may work from home up to 2 days per week.
        Only full-time employees are eligible after completing probation.
        Manager approval is mandatory before availing WFH.
        Emergency exceptions may be approved by HR.
        """,
        "metadata": {
            "category": "Work From Home Policy",
            "source": "Remote Work Guidelines"
        }
    },
    {
        "id": "appraisal_001",
        "text": """
        The appraisal cycle is conducted once every year in April.
        Employees are evaluated on a 5-point rating scale.
        Salary increments are linked to performance ratings and company budget.
        Managers discuss performance feedback during appraisal meetings.
        """,
        "metadata": {
            "category": "Appraisal Policy",
            "source": "Performance Handbook"
        }
    },
    {
        "id": "conduct_001",
        "text": """
        Employees must maintain respectful workplace behavior at all times.
        Confidential company and customer data must not be shared externally.
        Any conflict of interest must be disclosed immediately.
        Violations may result in disciplinary action.
        """,
        "metadata": {
            "category": "Code of Conduct",
            "source": "Employee Ethics Manual"
        }
    }
]

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = "chroma_hr_policy_db"
COLLECTION_NAME = "hr_policy_collection"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# FUNCTION 1
# -----------------------------
def create_embeddings(texts):
    """
    Convert list of texts into embedding vectors
    """
    return embedding_model.encode(texts).tolist()


# -----------------------------
# FUNCTION 2
# -----------------------------
def setup_vector_database():
    """
    Create persistent ChromaDB client + collection
    """
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


# -----------------------------
# FUNCTION 3
# -----------------------------
def index_hr_documents(collection):
    """
    Insert/update docs using UPSERT
    """
    texts = [doc["text"] for doc in HR_POLICY_DOCUMENTS]
    embeddings = create_embeddings(texts)

    collection.upsert(
        ids=[doc["id"] for doc in HR_POLICY_DOCUMENTS],
        documents=texts,
        metadatas=[doc["metadata"] for doc in HR_POLICY_DOCUMENTS],
        embeddings=embeddings
    )

    print("Documents indexed successfully.\n")


# -----------------------------
# FUNCTION 4
# -----------------------------
def retrieve_hr_content(collection, query, top_k=3):
    """
    Retrieve top matching chunks
    """
    query_embedding = create_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    chunks = []

    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return chunks


# -----------------------------
# FUNCTION 5
# -----------------------------
def build_grounded_prompt(query, chunks):
    """
    Build RAG prompt
    """
    context = ""

    for i, chunk in enumerate(chunks, 1):
        context += f"""
Chunk {i}:
Category: {chunk['metadata']['category']}
Source: {chunk['metadata']['source']}
Text: {chunk['text']}
"""

    prompt = f"""
You are an HR Policy Assistant.

Answer ONLY using the policy context below.

If the answer is not present in the context,
reply exactly:
"I could not find this information in the HR policy documents."

POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{query}
"""

    return prompt


# -----------------------------
# FUNCTION 6
# -----------------------------
def generate_answer(query, chunks):
    """
    Generate grounded answer using Gemini
    """
    prompt = build_grounded_prompt(query, chunks)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text


# -----------------------------
# FUNCTION 7
# -----------------------------
def answer_with_rag(collection, query, top_k=3):
    """
    Full RAG pipeline
    """
    print("=" * 60)
    print("QUERY:", query)

    chunks = retrieve_hr_content(collection, query, top_k)

    print("\nRetrieved Chunks:\n")

    for c in chunks:
        print(f"Category: {c['metadata']['category']}")
        print(f"Distance: {c['distance']:.4f}")
        print(c["text"])
        print("-" * 40)

    answer = generate_answer(query, chunks)

    print("\nRAG ANSWER:")
    print(answer)
    print("=" * 60)


# -----------------------------
# WITHOUT RAG
# -----------------------------
def generate_answer_without_retrieval(query):
    """
    LLM answers from memory only
    """

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"Answer this HR question: {query}"
    )

    return response.text


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    collection = setup_vector_database()

    index_hr_documents(collection)

    queries = [
        "How many days of annual leave am I entitled to per year?",
        "Do I need manager approval before working from home?",
        "When is the appraisal cycle conducted and how is increment decided?"
    ]

    for q in queries:
        answer_with_rag(collection, q)

    # side-by-side comparison
    test_query = "How many annual leave days do employees get?"

    print("\n" + "#" * 60)
    print("WITHOUT RAG")
    print(generate_answer_without_retrieval(test_query))

    print("\nWITH RAG")
    answer_with_rag(collection, test_query)