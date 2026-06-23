"""
src/rag/fashion_rag.py
-----------------------
Phase 8: Fashion Knowledge RAG (Retrieval-Augmented Generation).

Uses the 25 expert stylist rationales from outfits.csv as a
Fashion Knowledge Base.

When generating an explanation:
  1. Embed the user query + outfit context
  2. Retrieve top-k most relevant stylist rationales
  3. Feed as context to Gemini
  4. Gemini generates a grounded, expert-quality explanation

This prevents hallucination and grounds all reasoning in real
fashion expertise from the dataset.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RAG_INDEX_PATH = ROOT / "embeddings" / "rag_index.faiss"
RAG_DATA_PATH  = ROOT / "embeddings" / "rag_data.json"


# ══════════════════════════════════════════════════════════════════════════════
# Fashion RAG System
# ══════════════════════════════════════════════════════════════════════════════

class FashionRAG:
    """
    Lightweight RAG over the 25 expert stylist rationales.

    Uses sentence-transformers (all-MiniLM-L6-v2) for rationale embeddings
    because it's optimised for semantic text similarity — better than CLIP
    for retrieving relevant text passages.

    At inference time, retrieves the most semantically relevant rationales
    and provides them as grounding context to Gemini.
    """

    # SentenceTransformer model for RAG (lightweight, semantic-focused)
    ST_MODEL_ID = "all-MiniLM-L6-v2"

    def __init__(self):
        self.rationales = []     # List of rationale dicts
        self.embeddings = None   # (N, dim) rationale embeddings
        self.index      = None   # FAISS index
        self._model     = None   # lazy-loaded SentenceTransformer
        self._built     = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, outfits_df, force_rebuild: bool = False) -> None:
        """
        Build the RAG knowledge base from outfits_df.
        Caches embeddings to disk for fast loading.
        """
        import faiss

        if not force_rebuild and RAG_INDEX_PATH.exists() and RAG_DATA_PATH.exists():
            print("[RAG] Cache found — loading...")
            self._load_cache()
            return

        print("[RAG] Building Fashion Knowledge RAG...")

        # ── Prepare rationale documents ────────────────────────────────────
        # Each "document" = enriched rationale with outfit metadata for context
        self.rationales = []
        for _, row in outfits_df.iterrows():
            rationale = str(row.get("stylist_rationale", "")).strip()
            if not rationale:
                continue

            # Build a rich context string combining rationale + outfit metadata
            context = (
                f"Outfit: {row.get('theme', '')} | "
                f"Occasion: {row.get('occasion', '')} | "
                f"Gender: {row.get('gender', '')} | "
                f"Wear Type: {row.get('wear_type', '')} | "
                f"Palette: {row.get('palette', '')} | "
                f"Rationale: {rationale}"
            )

            self.rationales.append({
                "outfit_id": str(row.get("outfit_id", "")),
                "theme": str(row.get("theme", "")),
                "occasion": str(row.get("occasion", "")),
                "gender": str(row.get("gender", "")),
                "wear_type": str(row.get("wear_type", "")),
                "palette": str(row.get("palette", "")),
                "rationale": rationale,
                "context_str": context,
                "hero": str(row.get("hero", "")),
                "items_count": int(row.get("items_count", 0))
                    if str(row.get("items_count", "")).isdigit() else 0,
            })

        print(f"[RAG] {len(self.rationales)} rationale documents prepared")

        # ── Embed rationales ────────────────────────────────────────────────
        model = self._get_model()
        texts = [r["context_str"] for r in self.rationales]
        self.embeddings = model.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=True)
        self.embeddings = self.embeddings.astype(np.float32)

        # ── Build FAISS index ───────────────────────────────────────────────
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        self._built = True
        print(f"[RAG] ✓ Index built — {self.index.ntotal} rationales, dim={dim}")

        # ── Save cache ──────────────────────────────────────────────────────
        self._save_cache()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 3,
                 occasion_filter: Optional[str] = None,
                 gender_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve the top-k most relevant stylist rationales for a query.

        The query can be:
          - The user's original request ("office outfit for women")
          - An outfit description ("navy shirt + olive trousers + brown loafers")
          - A combination of both

        Returns list of rationale dicts with similarity scores.
        """
        if not self._built:
            raise RuntimeError("RAG not built. Call build() first.")

        # Embed query
        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

        # Search
        scores, indices = self.index.search(q_vec, min(k * 3, len(self.rationales)))
        scores, indices = scores[0], indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            r = self.rationales[idx].copy()
            r["similarity"] = float(score)

            # Optional filters
            if occasion_filter and r["occasion"] != occasion_filter:
                continue
            if gender_filter and r["gender"] != gender_filter:
                continue

            results.append(r)
            if len(results) >= k:
                break

        # If filters left us with < k, pad without filters
        if len(results) < k:
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                r = self.rationales[idx].copy()
                r["similarity"] = float(score)
                if r not in results:
                    results.append(r)
                if len(results) >= k:
                    break

        return results[:k]

    def format_context_for_llm(self, rationales: List[Dict]) -> str:
        """
        Format retrieved rationales into a structured context block
        for the Gemini prompt.
        """
        lines = ["=== Expert Stylist Knowledge Base ==="]
        for i, r in enumerate(rationales, 1):
            lines.append(
                f"\n[Expert {i} — {r['theme']}]\n"
                f"Occasion: {r['occasion']} | Gender: {r['gender']} | "
                f"Palette: {r['palette']}\n"
                f"Rationale: {r['rationale']}"
            )
        return "\n".join(lines)

    # ── Model + cache helpers ─────────────────────────────────────────────────

    def _get_model(self):
        """Lazy-load SentenceTransformer."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[RAG] Loading sentence-transformer: {self.ST_MODEL_ID}")
            self._model = SentenceTransformer(self.ST_MODEL_ID)
        return self._model

    def _save_cache(self) -> None:
        import faiss
        faiss.write_index(self.index, str(RAG_INDEX_PATH))
        with open(RAG_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "rationales": self.rationales,
                "model_id": self.ST_MODEL_ID,
            }, f, indent=2, ensure_ascii=False)
        print(f"[RAG] ✓ Cache saved → {RAG_INDEX_PATH}")

    def _load_cache(self) -> None:
        import faiss
        import numpy as np

        self.index = faiss.read_index(str(RAG_INDEX_PATH))

        with open(RAG_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.rationales = data["rationales"]

        # Re-encode to get embeddings array (needed for some ops)
        # Only if we actually need it — skip for retrieval-only mode
        self._built = True
        print(f"[RAG] ✓ Loaded {len(self.rationales)} rationales, "
              f"{self.index.ntotal} vectors")


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_rag(outfits_df, force_rebuild: bool = False) -> FashionRAG:
    """Build and return a FashionRAG instance."""
    rag = FashionRAG()
    rag.build(outfits_df, force_rebuild=force_rebuild)
    return rag


def load_rag(outfits_df=None) -> FashionRAG:
    """Load RAG from cache, or build if cache missing."""
    rag = FashionRAG()
    if RAG_INDEX_PATH.exists() and RAG_DATA_PATH.exists():
        rag._load_cache()
    elif outfits_df is not None:
        rag.build(outfits_df)
    else:
        raise FileNotFoundError("RAG cache not found. Pass outfits_df to build.")
    return rag


if __name__ == "__main__":
    from src.data.loader import load_outfits

    outfits = load_outfits()
    rag = build_rag(outfits, force_rebuild=True)

    # Test retrieval
    queries = [
        "office formal outfit for men",
        "party dress women evening heels",
        "ethnic kurta festive occasion",
    ]
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = rag.retrieve(q, k=2)
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['theme']}: {r['rationale'][:80]}...")
