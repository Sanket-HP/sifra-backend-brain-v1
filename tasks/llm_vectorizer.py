# ============================================================
#   SIFRA LLM Vectorizer v5.0  (COGNITIVE RAG + HUMAN CONTEXT)
#
#   NEW IN v5.0:
#     ✔ Improved semantic cleaning (human-style context retention)
#     ✔ Stronger CRE semantic boosting (v2)
#     ✔ Question-type detection (what/why/how intent)
#     ✔ Better TF-IDF → BM25++ hybrid scoring
#     ✔ Improved fallback ranking for short prompts
#     ✔ Regex FIX (no :-/ crash)
# ============================================================

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CLEAN TEXT — Human & Cognitive Friendly
# ============================================================
def clean_text(text: str) -> str:
    """
    Clean text while preserving punctuation needed for reasoning.
    FIXED: No invalid regex ranges.
    """

    text = str(text).lower().strip()

    # Safe regex: keep ., : , -, _, /, %
    text = re.sub(r"[^a-z0-9\s,\.\:\-\/%\_]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# BUILD VECTOR STORE
# ============================================================
def build_vector_store(documents: list):

    cleaned_docs = [clean_text(doc) for doc in documents if str(doc).strip()]

    if not cleaned_docs:
        return {
            "docs": [],
            "vectorizer": None,
            "matrix": None,
            "doc_len": [],
        }

    try:
        vectorizer = TfidfVectorizer(
            max_features=9000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(cleaned_docs)

    except Exception:
        # Failsafe mode
        vectorizer = None
        matrix = None

    # Track document lengths (used by BM25++)
    doc_len = np.array([len(doc.split()) for doc in cleaned_docs])

    return {
        "docs": cleaned_docs,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "doc_len": doc_len,
    }


# ============================================================
# BM25++ — Improved Weighting
# ============================================================
def _bm25pp(tfidf_scores, doc_lengths, avg_dl, k=1.6, b=0.78):
    """
    Enhanced BM25++ scoring for better reasoning chunks.
    """
    output = []
    for score, dl in zip(tfidf_scores, doc_lengths):
        denom = score + k * (1 - b + b * (dl / (avg_dl + 1e-7)))
        weight = (score * (k + 1)) / (denom + 1e-7)
        output.append(weight)
    return np.array(output)


# ============================================================
# CRE BOOSTER v2 — Semantic + Intent + Structure
# ============================================================
def _cre_boost(scores, docs, query):

    query_words = set(clean_text(query).split())
    boosted = []

    for score, doc in zip(scores, docs):
        base = score

        # -----------------------------
        # 1) Keyword overlap boost
        # -----------------------------
        doc_words = set(doc.split())
        base += len(query_words.intersection(doc_words)) * 0.03

        # -----------------------------
        # 2) Deep reasoning punctuation
        # -----------------------------
        structure = doc.count(",") + doc.count(".") + doc.count(":")
        base += structure * 0.005

        # -----------------------------
        # 3) Intent awareness boost (why/how/what)
        # -----------------------------
        if query.startswith("why"):
            base += doc.count("because") * 0.05

        if query.startswith("how"):
            base += doc.count("steps") * 0.04

        if query.startswith("what"):
            base += doc.count("means") * 0.02

        boosted.append(base)

    return np.array(boosted)


# ============================================================
# SEARCH ENGINE (TF-IDF + BM25++ + CRE Hybrid)
# ============================================================
def search_vector_store(vector_store: dict, query: str, top_k=3):

    try:
        query = clean_text(query)

        docs = vector_store.get("docs", [])
        vectorizer = vector_store.get("vectorizer")
        matrix = vector_store.get("matrix")
        doc_len = vector_store.get("doc_len")

        if len(docs) == 0:
            return ["No documents available in knowledge base."]

        # ======================================================
        # FULL MODE — TF-IDF + BM25++ + CRE
        # ======================================================
        if vectorizer is not None and matrix is not None:

            query_vec = vectorizer.transform([query])
            tfidf_scores = (matrix @ query_vec.T).toarray().flatten()

            # No match → fallback to longest docs
            if tfidf_scores.max() <= 0:
                top_ids = np.argsort(doc_len)[::-1][:top_k]
                return [docs[i] for i in top_ids]

            avg_dl = doc_len.mean()

            bm25_scores = _bm25pp(tfidf_scores, doc_len, avg_dl)
            hybrid = (tfidf_scores * 0.55) + (bm25_scores * 0.45)

            cre_scores = _cre_boost(hybrid, docs, query)

            ranked = np.argsort(cre_scores)[::-1][:top_k]
            return [docs[i] for i in ranked]

        # ======================================================
        # FALLBACK MODE (keyword match only)
        # ======================================================
        q_words = query.split()

        scores = [
            (sum(w in doc for w in q_words), i)
            for i, doc in enumerate(docs)
        ]

        scores = sorted(scores, reverse=True)
        top_ids = [idx for _, idx in scores[:top_k]]

        return [docs[i] for i in top_ids]

    except Exception as e:
        return [f"[SAFE MODE] Search failed: {str(e)}"]
