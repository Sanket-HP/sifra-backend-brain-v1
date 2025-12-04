# ============================================================
#   SIFRA LLM Vectorizer v2.1 (NO CIRCULAR IMPORTS)
#   TF-IDF + BM25-Lite Hybrid
#   JSON SAFE, FAST & STABLE
# ============================================================

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CLEAN TEXT
# ============================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s,.:-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# BUILD VECTOR STORE  (NO SELF IMPORT)
# ============================================================
def build_vector_store(documents: list):
    cleaned_docs = [clean_text(doc) for doc in documents if str(doc).strip()]

    if len(cleaned_docs) == 0:
        return {
            "docs": [],
            "vectors": [],
            "vectorizer": None,
            "matrix": None,
            "doc_len": []
        }

    try:
        vectorizer = TfidfVectorizer(
            max_features=7000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(cleaned_docs)
        dense_vectors = matrix.toarray().tolist()

    except Exception:
        dense_vectors = np.random.rand(len(cleaned_docs), 32).tolist()
        vectorizer = None
        matrix = None

    doc_len = np.array([len(doc.split()) for doc in cleaned_docs])

    return {
        "docs": cleaned_docs,
        "vectors": dense_vectors,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "doc_len": doc_len,
    }


# ============================================================
# BM25-LITE BOOSTER
# ============================================================
def _bm25lite(tfidf_scores, doc_lengths, avg_dl, k=1.2, b=0.75):
    adjusted_scores = []
    for score, dl in zip(tfidf_scores, doc_lengths):
        denom = score + k * (1 - b + b * (dl / (avg_dl + 1e-7)))
        w = score * (k + 1) / (denom + 1e-7)
        adjusted_scores.append(w)
    return np.array(adjusted_scores)


# ============================================================
# SEARCH (NO CIRCULAR IMPORT)
# ============================================================
def search_vector_store(vector_store: dict, query: str, top_k=3):

    try:
        query = clean_text(query)

        docs = vector_store.get("docs", [])
        vectors = vector_store.get("vectors", [])

        if len(docs) == 0:
            return ["No documents available in knowledge base."]

        vectorizer = vector_store.get("vectorizer")
        matrix = vector_store.get("matrix")
        doc_lengths = vector_store.get("doc_len")

        if vectorizer is not None and matrix is not None:

            query_vec = vectorizer.transform([query])
            tfidf_scores = (matrix @ query_vec.T).toarray().flatten()

            if tfidf_scores.max() <= 0:
                longest_ids = np.argsort(doc_lengths)[::-1][:top_k]
                return [docs[i] for i in longest_ids]

            avg_dl = doc_lengths.mean()
            bm25_scores = _bm25lite(tfidf_scores, doc_lengths, avg_dl)

            final_scores = (tfidf_scores * 0.6) + (bm25_scores * 0.4)

            top_ids = np.argsort(final_scores)[::-1][:top_k]

            return [docs[i] for i in top_ids]

        # fallback matching
        q_words = clean_text(query).split()
        scores = [(sum(1 for w in q_words if w in d), i) for i, d in enumerate(docs)]
        scores = sorted(scores, reverse=True)
        top_ids = [i for _, i in scores[:top_k]]

        return [docs[i] for i in top_ids]

    except Exception as e:
        return [f"Vector search failed safely: {str(e)}"]
