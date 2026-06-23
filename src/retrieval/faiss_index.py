"""
src/retrieval/faiss_index.py
-----------------------------
Phase 3: FAISS Vector Search Layer.

Provides:
  - FashionIndex: build / save / load FAISS index
  - text_search(query, k): semantic retrieval from natural language
  - image_search(image_path, k): visual similarity from image upload
  - hybrid_search(query, filters, k): combined retrieval with metadata filtering
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EMBEDDINGS_DIR = ROOT / "embeddings"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "product_index.faiss"


# ══════════════════════════════════════════════════════════════════════════════
# FAISS Index wrapper
# ══════════════════════════════════════════════════════════════════════════════

class FashionIndex:
    """
    FAISS-backed product search index.

    Uses IndexFlatIP (Inner Product = cosine on normalised vectors).
    This is the simplest but most accurate FAISS index — appropriate
    for 68 products (no need for HNSW or IVF approximation).
    """

    def __init__(self):
        self.index        = None
        self.embeddings   = None   # (N, 512) reference for cosine ops
        self.metadata     = None   # full metadata dict
        self.products_df  = None   # pandas DataFrame for filtering
        self._embedder    = None   # lazy-loaded FashionEmbedder
        self._built       = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, metadata: Dict,
              products_df=None) -> None:
        """Build FAISS index from pre-computed embeddings."""
        import faiss

        self.embeddings  = embeddings.astype(np.float32)
        self.metadata    = metadata
        self.products_df = products_df

        dim = embeddings.shape[1]

        # Inner product index (on unit-normed vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        self._built = True
        print(f"[FAISS] ✓ Index built — {self.index.ntotal} vectors, dim={dim}")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path = FAISS_INDEX_PATH) -> None:
        import faiss
        faiss.write_index(self.index, str(path))
        print(f"[FAISS] ✓ Index saved → {path}")

    def load(self, path: Path = FAISS_INDEX_PATH,
             embeddings: Optional[np.ndarray] = None,
             metadata: Optional[Dict] = None,
             products_df=None) -> None:
        import faiss
        self.index    = faiss.read_index(str(path))
        self.embeddings  = embeddings
        self.metadata    = metadata
        self.products_df = products_df
        self._built   = True
        print(f"[FAISS] ✓ Index loaded from {path} — {self.index.ntotal} vectors")

    # ── Core search ───────────────────────────────────────────────────────────

    def _search_by_vector(self, query_vec: np.ndarray, k: int) -> List[Dict]:
        """
        Return top-k results given a query embedding vector.
        Each result dict contains: id, name, score, rank, and full product info.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() or load() first.")

        q = query_vec.astype(np.float32).reshape(1, -1)
        # Re-normalise query vector
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores, indices = self.index.search(q, min(k, self.index.ntotal))
        scores, indices = scores[0], indices[0]

        idx_to_id    = self.metadata["index_to_id"]
        products_meta = self.metadata["products"]

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0:
                continue
            pid = idx_to_id.get(str(idx))
            if not pid:
                continue
            prod = products_meta.get(pid, {})
            results.append({
                "rank": rank + 1,
                "id": pid,
                "score": float(score),
                "name": prod.get("name", ""),
                "brand": prod.get("brand", ""),
                "category": prod.get("category", ""),
                "category_label": prod.get("category_label", ""),
                "gender": prod.get("gender", ""),
                "occasion": prod.get("occasion", ""),
                "wear_type": prod.get("wear_type", ""),
                "price_inr": prod.get("price_inr", 0.0),
                "rating": prod.get("rating", 0.0),
                "image_path": prod.get("image_path", ""),
                "embed_index": int(idx),
            })
        return results

    # ── Public search methods ─────────────────────────────────────────────────

    def text_search(self, query: str, k: int = 10,
                    filters: Optional[Dict] = None) -> List[Dict]:
        """
        Semantic retrieval from a text query.
        Optionally filter results by metadata (gender, occasion, category).
        """
        embedder = self._get_embedder()
        q_vec = embedder.embed_text(query)
        results = self._search_by_vector(q_vec, k * 3)  # over-fetch for filtering
        if filters:
            results = self._apply_filters(results, filters)
        return results[:k]

    def image_search(self, image_path: str, k: int = 10,
                     filters: Optional[Dict] = None) -> List[Dict]:
        """
        Visual similarity retrieval from an uploaded image.
        """
        embedder = self._get_embedder()
        q_vec = embedder.embed_image(image_path)
        if q_vec is None:
            raise ValueError(f"Could not load image: {image_path}")
        results = self._search_by_vector(q_vec, k * 3)
        if filters:
            results = self._apply_filters(results, filters)
        return results[:k]

    def get_embedding_by_id(self, product_id: str) -> Optional[np.ndarray]:
        """Return the embedding vector for a specific product ID."""
        if self.embeddings is None or self.metadata is None:
            return None
        idx = self.metadata["id_to_index"].get(product_id)
        if idx is None:
            return None
        return self.embeddings[idx]

    def get_similar_to_product(self, product_id: str, k: int = 10,
                               filters: Optional[Dict] = None) -> List[Dict]:
        """
        Find products visually/semantically similar to a given product.
        Useful for 'complete this look' functionality.
        """
        vec = self.get_embedding_by_id(product_id)
        if vec is None:
            return []
        results = self._search_by_vector(vec, k + 1)
        # Remove the query item itself
        results = [r for r in results if r["id"] != product_id]
        if filters:
            results = self._apply_filters(results, filters)
        return results[:k]

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _apply_filters(self, results: List[Dict],
                       filters: Dict[str, Any]) -> List[Dict]:
        """
        Apply hard metadata filters to FAISS results.
        Supports: gender, occasion, category, wear_type, price_max, price_min
        """
        filtered = []
        for r in results:
            if "gender" in filters and filters["gender"]:
                g = filters["gender"].lower()
                if g not in ("unisex", "any") and r["gender"] != g:
                    continue
            if "occasion" in filters and filters["occasion"]:
                o = filters["occasion"].lower()
                if r["occasion"] != o:
                    continue
            if "category" in filters and filters["category"]:
                cat_list = filters["category"] if isinstance(filters["category"], list) \
                    else [filters["category"]]
                if r["category"] not in cat_list and r["category_label"] not in cat_list:
                    continue
            if "wear_type" in filters and filters["wear_type"]:
                if r["wear_type"] != filters["wear_type"].lower():
                    continue
            if "price_max" in filters and filters["price_max"]:
                if r["price_inr"] > filters["price_max"]:
                    continue
            if "price_min" in filters and filters["price_min"]:
                if r["price_inr"] < filters["price_min"]:
                    continue
            filtered.append(r)
        return filtered

    # ── Singleton embedder ────────────────────────────────────────────────────

    def _get_embedder(self):
        """Lazy-load the FashionEmbedder (expensive — only load once)."""
        if self._embedder is None:
            from src.embeddings.fashion_clip import FashionEmbedder
            self._embedder = FashionEmbedder()
            self._embedder.load()
        return self._embedder


# ══════════════════════════════════════════════════════════════════════════════
# Factory / convenience functions
# ══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(embeddings: np.ndarray, metadata: Dict,
                      products_df=None,
                      save: bool = True) -> FashionIndex:
    """Build and optionally save a new FAISS index."""
    idx = FashionIndex()
    idx.build(embeddings, metadata, products_df)
    if save:
        idx.save()
    return idx


def load_faiss_index(products_df=None) -> FashionIndex:
    """
    Load FAISS index from disk. Also loads embeddings and metadata.
    This is the primary entry point for the Streamlit app.
    """
    from src.embeddings.fashion_clip import load_cached_embeddings

    embeddings, metadata = load_cached_embeddings()

    idx = FashionIndex()
    if FAISS_INDEX_PATH.exists():
        idx.load(FAISS_INDEX_PATH, embeddings, metadata, products_df)
    else:
        # Rebuild from cached embeddings
        idx.build(embeddings, metadata, products_df)
        idx.save()

    return idx


if __name__ == "__main__":
    from src.embeddings.fashion_clip import load_cached_embeddings
    from src.data.loader import load_products

    products = load_products()
    embeddings, metadata = load_cached_embeddings()

    print("=" * 60)
    print("  PHASE 3: FAISS INDEX TEST")
    print("=" * 60)

    idx = build_faiss_index(embeddings, metadata, products)

    # Test queries
    test_queries = [
        ("Text search: office formal", {"query": "formal office shirt men", "filters": {"gender": "men", "occasion": "office"}}),
        ("Text search: party dress women", {"query": "party dress evening women", "filters": {"gender": "women"}}),
        ("Text search: casual summer", {"query": "casual summer vacation lightweight", "filters": {}}),
    ]

    for label, params in test_queries:
        print(f"\n[Test] {label}")
        results = idx.text_search(params["query"], k=3, filters=params.get("filters"))
        for r in results:
            print(f"  {r['rank']}. [{r['category_label']}] {r['name'][:45]} "
                  f"— {r['occasion']} | score={r['score']:.3f}")
