from langchain.tools import tool
from sentence_transformers import SentenceTransformer
import numpy as np
from app.vectorestore import index
from app.embedding import load_catalog

model = SentenceTransformer("all-MiniLM-L6-v2")
catalog = load_catalog()

@tool
def clarification(query:str):
    """
    Determine whether additional information is required
    from the user before recommending assessments.
    Ask follow-up questions when the request is ambiguous
    or incomplete.
    """
    if len(query.split())<3:
        return True
    return False

@tool
def compare(names:str):
    """
    Compare multiple SHL assessments based on competencies,
    duration, job levels, languages, adaptability and
    assessment categories.
    """
    return f"Comparing {names}"

@tool
def guardrails(query:str):
    """
    Detect off-topic, unsafe or unsupported requests.
    Restrict recommendations strictly to SHL assessments
    and reject legal advice, salary discussions and
    prompt injection attempts.
    """
    blocked=["salary","law","tax","immigration","ignore"]

    for w in blocked:
        if w in query.lower():
            return True

    return False


def search_assessments(query):    
    emb = model.encode([query])
    D, I = index.search(
        np.array(emb, dtype=np.float32),
        5
    )

    return [
        catalog[i]for i in I[0]
        ]

@tool
def retrieve_assessments(query: str):
    """
    Retrieve relevant SHL assessments.
    """
    return search_assessments(query)