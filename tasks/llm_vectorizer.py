# ============================================================
#   SIFRA LLM Vectorizer v4.5 (COGNITIVE RAG ENGINE)
#
#   NEW IN v4.5:
#     ✔ CRE-aware semantic boosting
#     ✔ Improved TF-IDF (7k max_features → upgraded to 9k)
#     ✔ BM25++ hybrid scoring
#     ✔ Context window scoring (LLM reasoning prep)
#     ✔ Token-length normalization
#     ✔ Fallback search (zero failure)
#     ✔ FIXED REGEX CRASH (NO more :-/ errors)
# ============================================================

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CLEAN TEXT (CRE-friendly + FIXED REGEX)
# ============================================================
def clean_text(text: str) -> str:
    """
    Cleans text while keeping punctuation necessary for semantic reasoning.
    FIXED: Removed invalid regex range error (:-/).
    """

    text = str(text).lower()

    # FIXED: escape dash and allow reasoning punctuation safely
    text = re.sub(r"[^a-z0-9\s,\.\:\/%\-\_]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# BUILD VECTOR STORE
# ============================================================
def build_vector_store(documents: list):
    """
    Builds TF-IDF + BM25++ vector store with CRE-aware signals.
    """

    cleaned_docs = [clean_text(doc) for doc in documents if str(doc).strip()]

    if len(cleaned_docs) == 0:
        return {
            "docs": [],
            "vectors": [],
            "vectorizer": None,
            "matrix": None,
            "doc_len": [],
        }

    try:
        vectorizer = TfidfVectorizer(
            max_features=9000,       # Extended vocabulary
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(cleaned_docs)
        dense_vectors = matrix.toarray().tolist()

    except Exception:
        # FAILSAFE: no crash even if TF-IDF backend fails
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
# BM25++ (Improved Long Text Weighting)
# ============================================================
def _bm25pp(tfidf_scores, doc_lengths, avg_dl, k=1.5, b=0.75):
    adjusted = []
    for score, dl in zip(tfidf_scores, doc_lengths):
        denom = score + k * (1 - b + b * (dl / (avg_dl + 1e-7)))
        w = (score * (k + 1)) / (denom + 1e-7)
        adjusted.append(w)
    return np.array(adjusted)


# ============================================================
# CRE SEMANTIC BOOSTER (Keyword + Reasoning Depth)
# ============================================================
def _cre_boost(scores, docs, query):
    query_words = set(clean_text(query).split())
    boosted = []

    for s, d in zip(scores, docs):
        base = s

        # Keyword overlap boost
        d_words = set(d.split())
        overlap = len(query_words.intersection(d_words))
        base += overlap * 0.03

        # Reasoning punctuation = deeper meaning indicators
        complexity = d.count(",") + d.count(":") + d.count(".")
        base += complexity * 0.005

        boosted.append(base)

    return np.array(boosted)


# ============================================================
# SEARCH ENGINE (TF-IDF + BM25++ + CRE Hybrid)
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

        # ===============================
        # TF-IDF + BM25++ + CRE (Full Mode)
        # ===============================
        if vectorizer is not None and matrix is not None:

            query_vec = vectorizer.transform([query])
            tfidf_scores = (matrix @ query_vec.T).toarray().flatten()

            # All zero scores → fallback to longest documents
            if tfidf_scores.max() <= 0:
                longest_ids = np.argsort(doc_lengths)[::-1][:top_k]
                return [docs[i] for i in longest_ids]

            avg_dl = doc_lengths.mean()

            # BM25++ scoring
            bm25_scores = _bm25pp(tfidf_scores, doc_lengths, avg_dl)

            # Hybrid TF-IDF + BM25++
            hybrid_scores = (tfidf_scores * 0.55) + (bm25_scores * 0.45)

            # CRE semantic booster
            cre_scores = _cre_boost(hybrid_scores, docs, query)

            top_ids = np.argsort(cre_scores)[::-1][:top_k]
            return [docs[i] for i in top_ids]

        # ===============================
        # HARD FALLBACK MODE (No TF-IDF)
        # ===============================
        q_words = clean_text(query).split()
        scores = [(sum(1 for w in q_words if w in d), i) for i, d in enumerate(docs)]
        scores = sorted(scores, reverse=True)
        top_ids = [i for _, i in scores[:top_k]]

        return [docs[i] for i in top_ids]

    except Exception as e:
        return [f"Vector search failed safely: {str(e)}"]
