"""
Vector RAG Engine — Upgrade #1
Replaces TF-IDF keyword search with:
  • sentence-transformers embeddings (all-MiniLM-L6-v2)
  • ChromaDB persistent vector store
  • Semantic cosine similarity search
Falls back gracefully to TF-IDF if chromadb/sentence-transformers not installed.
"""
import os
import logging

GUIDELINE_DIR = "medical_guidelines"
CHROMA_DIR = "chroma_db"

logger = logging.getLogger(__name__)

# ── Try to load vector stack ──────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = _chroma_client.get_or_create_collection(
        name="medical_guidelines",
        metadata={"hnsw:space": "cosine"},
    )
    VECTOR_RAG_AVAILABLE = True
    logger.info("✅ Vector RAG ready (ChromaDB + sentence-transformers)")
except Exception as _e:
    VECTOR_RAG_AVAILABLE = False
    logger.warning(f"⚠️  Vector RAG unavailable ({_e}), falling back to TF-IDF")


# ── Document loader ───────────────────────────────────────────────────────────
def load_documents():
    docs = []
    if not os.path.isdir(GUIDELINE_DIR):
        return docs
    for filename in os.listdir(GUIDELINE_DIR):
        path = os.path.join(GUIDELINE_DIR, filename)
        if filename.endswith(".txt"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs.append({"source": filename, "text": text})
            except Exception as e:
                logger.warning(f"Error reading text file {filename}: {e}")
        elif filename.endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(path)
                text = ""
                for page in doc:
                    text += page.get_text()
                docs.append({"source": filename, "text": text})
            except Exception as e:
                logger.warning(f"Error reading PDF file {filename}: {e}")
    return docs



# ── Index documents into ChromaDB ────────────────────────────────────────────
def _index_documents():
    """Index all guideline docs into ChromaDB (idempotent)."""
    if not VECTOR_RAG_AVAILABLE:
        return
    docs = load_documents()
    if not docs:
        return
    existing = set(_collection.get()["ids"])
    to_add = [d for d in docs if d["source"] not in existing]
    if not to_add:
        return
    embeddings = _st_model.encode([d["text"] for d in to_add]).tolist()
    _collection.add(
        ids=[d["source"] for d in to_add],
        embeddings=embeddings,
        documents=[d["text"] for d in to_add],
        metadatas=[{"source": d["source"]} for d in to_add],
    )
    logger.info(f"✅ Indexed {len(to_add)} medical guideline docs into ChromaDB")


# Index on import
_index_documents()


# ── Semantic search ───────────────────────────────────────────────────────────
def retrieve_guidelines(query: str, top_k: int = 3):
    """Semantic similarity search over medical guidelines."""
    if VECTOR_RAG_AVAILABLE:
        return _vector_search(query, top_k)
    return _tfidf_search(query, top_k)


def _vector_search(query: str, top_k: int):
    try:
        query_embedding = _st_model.encode([query]).tolist()
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, max(_collection.count(), 1)),
        )
        out = []
        for i, doc_text in enumerate(results["documents"][0]):
            out.append({
                "source": results["metadatas"][0][i]["source"],
                "score": round(1 - results["distances"][0][i], 3),
                "text": doc_text[:500],
                "method": "semantic",
            })
        return out
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return _tfidf_search(query, top_k)


def _tfidf_search(query: str, top_k: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = load_documents()
    if not docs:
        return []
    texts = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [
        {"source": d["source"], "score": round(float(s), 3), "text": d["text"][:500], "method": "tfidf"}
        for d, s in ranked[:top_k]
    ]


# ── Public API ────────────────────────────────────────────────────────────────
def rag_medical_explanation(symptoms: list, medical_history: list):
    query = " ".join(symptoms + medical_history)
    retrieved_docs = retrieve_guidelines(query)
    explanation_parts = [f"From {d['source']}: {d['text']}" for d in retrieved_docs]
    return {
        "query": query,
        "retrieved_guidelines": retrieved_docs,
        "rag_explanation": " ".join(explanation_parts) if explanation_parts else "No guideline found.",
        "method": "vector_semantic" if VECTOR_RAG_AVAILABLE else "tfidf_keyword",
    }
