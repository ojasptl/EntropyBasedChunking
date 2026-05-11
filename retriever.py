from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# BM25 index (built once after PDF is processed)
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self, corpus: List[str]):
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = corpus

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (corpus_index, score) pairs sorted by descending BM25 score."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# Hybrid retrieval: Qdrant (dense) + BM25 (sparse) → merge → dedup → rerank
# ---------------------------------------------------------------------------

def _normalize(scores: List[float]) -> List[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_search(query: str, collection_name: str,
                  embedding_model: SentenceTransformer,
                  qdrant_client: QdrantClient,
                  bm25_index: BM25Index,
                  top_k: int = 5,
                  dense_weight: float = 0.6,
                  bm25_weight: float = 0.4,
                  fetch_k: int = 15) -> List[dict]:
    """
    1. Dense retrieval via Qdrant
    2. Sparse retrieval via BM25
    3. Reciprocal Rank Fusion merge + dedup
    4. Cross-encoder rerank
    5. Return top_k results
    """

    # ── 1. Dense retrieval ──────────────────────────────────────────────────
    query_embedding = embedding_model.encode([query])[0]
    dense_hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=fetch_k,
        with_payload=True,
    ).points

    # ── 2. BM25 retrieval ───────────────────────────────────────────────────
    bm25_hits = bm25_index.search(query, top_k=fetch_k)  # [(corpus_idx, score)]

    # ── 3. Merge + dedup using Reciprocal Rank Fusion ───────────────────────
    # Build a dict keyed by text snippet to avoid duplicates
    candidates: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        text = hit.payload["text"]
        key = text[:200]           # use first 200 chars as dedup key
        rrf_dense = 1.0 / (60 + rank + 1)
        if key not in candidates:
            candidates[key] = {
                "text": text,
                "payload": hit.payload,
                "rrf_score": 0.0,
            }
        candidates[key]["rrf_score"] += dense_weight * rrf_dense

    for rank, (corpus_idx, _) in enumerate(bm25_hits):
        text = bm25_index.corpus[corpus_idx]
        key = text[:200]
        rrf_bm25 = 1.0 / (60 + rank + 1)
        if key not in candidates:
            candidates[key] = {
                "text": text,
                "payload": {"text": text, "chunk_idx": corpus_idx,
                            "sentence_count": 0, "chunk_length": len(text)},
                "rrf_score": 0.0,
            }
        candidates[key]["rrf_score"] += bm25_weight * rrf_bm25

    merged = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)

    # ── 4. Cross-encoder rerank ─────────────────────────────────────────────
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c["text"]) for c in merged]
        ce_scores = reranker.predict(pairs)
        for i, c in enumerate(merged):
            c["rerank_score"] = float(ce_scores[i])
        merged = sorted(merged, key=lambda x: x["rerank_score"], reverse=True)
    except Exception as e:
        print(f"⚠️  Reranker unavailable ({e}), using RRF scores only.")

    return merged[:top_k]


# ---------------------------------------------------------------------------
# Pretty-print results (called from main)
# ---------------------------------------------------------------------------

def print_results(results: List[dict], query: str) -> str:
    print(f"\n📑 Top {len(results)} Results:\n")
    print("=" * 80)
    for idx, result in enumerate(results, 1):
        payload = result["payload"]
        rerank = result.get("rerank_score", result.get("rrf_score", 0.0))
        print(f"\n[Result {idx}] (Rerank score: {rerank:.4f})")
        print(f"Chunk #{payload.get('chunk_idx', '?')} | "
              f"Sentences: {payload.get('sentence_count', '?')} | "
              f"Length: {payload.get('chunk_length', '?')}")
        print(f"\n{result['text']}")
        print("-" * 80)

    return "\n\n".join([r["text"] for r in results])