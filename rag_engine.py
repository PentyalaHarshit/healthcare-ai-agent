import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GUIDELINE_DIR = "medical_guidelines"


def load_documents():
    docs = []

    for filename in os.listdir(GUIDELINE_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(GUIDELINE_DIR, filename)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            docs.append({
                "source": filename,
                "text": text
            })

    return docs


def retrieve_guidelines(query: str, top_k: int = 2):
    docs = load_documents()

    if not docs:
        return []

    texts = [doc["text"] for doc in docs]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts + [query])

    query_vector = matrix[-1]
    doc_vectors = matrix[:-1]

    scores = cosine_similarity(query_vector, doc_vectors)[0]

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for doc, score in ranked[:top_k]:
        results.append({
            "source": doc["source"],
            "score": round(float(score), 3),
            "text": doc["text"]
        })

    return results


def rag_medical_explanation(symptoms, medical_history):
    query = " ".join(symptoms + medical_history)

    retrieved_docs = retrieve_guidelines(query)

    explanation_parts = []

    for doc in retrieved_docs:
        explanation_parts.append(
            f"From {doc['source']}: {doc['text']}"
        )

    return {
        "query": query,
        "retrieved_guidelines": retrieved_docs,
        "rag_explanation": (
            " ".join(explanation_parts)
            if explanation_parts
            else "No guideline document found."
        )
    }