"""
src/embeddings/fashion_clip.py
-------------------------------
Phase 2: Multi-modal Fashion Embeddings using FashionCLIP.

FashionCLIP (patrickjohncyh/fashion-clip) is a domain-specific CLIP model
fine-tuned on ~700K fashion product images. It understands fashion semantics
far better than generic CLIP.

Produces:
  - Image embeddings  (512-dim)
  - Text embeddings   (512-dim)
  - Combined embeddings = normalised mean of both (512-dim)

All embeddings cached to disk:
  embeddings/product_embeddings.npy   ← combined (68, 512)
  embeddings/image_embeddings.npy     ← image only (68, 512)
  embeddings/text_embeddings.npy      ← text only (68, 512)
  embeddings/product_metadata.json    ← id→index mapping + product info
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

EMBEDDINGS_DIR = ROOT / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Output files
COMBINED_PATH  = EMBEDDINGS_DIR / "product_embeddings.npy"
IMAGE_PATH     = EMBEDDINGS_DIR / "image_embeddings.npy"
TEXT_PATH      = EMBEDDINGS_DIR / "text_embeddings.npy"
METADATA_PATH  = EMBEDDINGS_DIR / "product_metadata.json"

# FashionCLIP model id — falls back to standard CLIP if unavailable
FASHION_CLIP_ID = "patrickjohncyh/fashion-clip"
CLIP_FALLBACK   = "openai/clip-vit-base-patch32"

EMBEDDING_DIM = 512  # both CLIP variants produce 512-d


# ══════════════════════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════════════════════

class FashionEmbedder:
    """
    Wraps FashionCLIP (or standard CLIP as fallback) and provides
    image + text embedding generation with on-disk caching.
    """

    def __init__(self, model_id: str = FASHION_CLIP_ID, device: str = "cpu"):
        self.model_id = model_id
        self.device   = device
        self.model    = None
        self.processor = None
        self._loaded   = False

    def load(self) -> None:
        """Lazy-load the model. Call once before embedding."""
        if self._loaded:
            return

        print(f"[FashionCLIP] Loading model: {self.model_id}")
        t0 = time.time()
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model     = CLIPModel.from_pretrained(self.model_id)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model.eval()
            self.model.to(self.device)
            elapsed = time.time() - t0
            print(f"[FashionCLIP] ✓ Loaded '{self.model_id}' in {elapsed:.1f}s")
        except Exception as e:
            print(f"[FashionCLIP] ✗ Failed to load '{self.model_id}': {e}")
            print(f"[FashionCLIP]   Falling back to: {CLIP_FALLBACK}")
            from transformers import CLIPModel, CLIPProcessor
            self.model     = CLIPModel.from_pretrained(CLIP_FALLBACK)
            self.processor = CLIPProcessor.from_pretrained(CLIP_FALLBACK)
            self.model.eval()
            self.model.to(self.device)
            self.model_id  = CLIP_FALLBACK
            print(f"[FashionCLIP] ✓ Fallback loaded in {time.time()-t0:.1f}s")

        self._loaded = True

    # ── Image embedding ───────────────────────────────────────────────────────

    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Encode a single product image.
        Returns normalised 512-d numpy array or None if image unreadable.
        """
        import torch
        self.load()
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            # Handle both tensor and BaseModelOutput return types
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output if hasattr(feats, 'pooler_output') else feats.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"  [WARN] Image embed failed for {image_path}: {e}")
            return None

    def embed_images_batch(self, image_paths: List[str],
                           batch_size: int = 8) -> np.ndarray:
        """
        Batch-encode images. Returns (N, 512) array.
        Failed images get a zero vector.
        """
        import torch
        self.load()
        n = len(image_paths)
        result = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

        for start in range(0, n, batch_size):
            batch_paths = image_paths[start:start + batch_size]
            imgs, valid_idx = [], []
            for i, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(img)
                    valid_idx.append(start + i)
                except Exception:
                    pass

            if not imgs:
                continue

            inputs = self.processor(images=imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output if hasattr(feats, 'pooler_output') else feats.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy().astype(np.float32)

            for j, idx in enumerate(valid_idx):
                if j < len(feats_np):
                    result[idx] = feats_np[j]

        return result

    # ── Text embedding ────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Batch-encode text strings. Returns (N, 512) array.
        """
        import torch
        self.load()
        n = len(texts)
        result = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            # CLIP max token length is 77; truncate silently
            batch_clean = [t[:300] for t in batch]
            inputs = self.processor(
                text=batch_clean, return_tensors="pt",
                padding=True, truncation=True, max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output if hasattr(feats, 'pooler_output') else feats.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy().astype(np.float32)
            result[start:start + len(batch)] = feats_np

        return result

    def embed_text(self, text: str) -> np.ndarray:
        """Single text embedding — used at query time."""
        return self.embed_texts([text])[0]

    def embed_image_file(self, image_path: str) -> Optional[np.ndarray]:
        """Single image embedding — used for image upload queries."""
        return self.embed_image(image_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main embedding pipeline
# ══════════════════════════════════════════════════════════════════════════════

def generate_product_embeddings(
    products_df,
    force_regenerate: bool = False,
    image_weight: float = 0.6,
    text_weight: float = 0.4,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate and cache multi-modal embeddings for all products.

    Embedding strategy:
      combined = normalise(image_weight × img_emb + text_weight × txt_emb)

    Image weight is higher (0.6) because FashionCLIP is trained on images
    and visual similarity is the most direct fashion signal.

    Returns:
      combined_embeddings: (68, 512) float32 array
      metadata: dict mapping product index → product info
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    if not force_regenerate and COMBINED_PATH.exists() and METADATA_PATH.exists():
        print("[Embeddings] Cache found — loading from disk...")
        combined = np.load(str(COMBINED_PATH))
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        print(f"[Embeddings] ✓ Loaded {len(combined)} cached embeddings ({combined.shape})")
        return combined, metadata

    # ── Fresh generation ──────────────────────────────────────────────────────
    print("[Embeddings] Generating embeddings for all products...")
    embedder = FashionEmbedder()
    embedder.load()

    n = len(products_df)
    image_paths = products_df["image_path"].tolist()
    embed_texts = products_df["embed_text"].tolist()

    # ── Image embeddings ──────────────────────────────────────────────────────
    print(f"\n[Embeddings] Step 1/3: Image embeddings ({n} products)...")
    t0 = time.time()
    image_embeddings = embedder.embed_images_batch(image_paths, batch_size=8)
    print(f"  ✓ Image embeddings done in {time.time()-t0:.1f}s — shape: {image_embeddings.shape}")

    # ── Text embeddings ───────────────────────────────────────────────────────
    print(f"\n[Embeddings] Step 2/3: Text embeddings ({n} products)...")
    t0 = time.time()
    text_embeddings = embedder.embed_texts(embed_texts, batch_size=16)
    print(f"  ✓ Text embeddings done in {time.time()-t0:.1f}s — shape: {text_embeddings.shape}")

    # ── Combined embedding ────────────────────────────────────────────────────
    print(f"\n[Embeddings] Step 3/3: Combining embeddings...")
    print(f"  Image weight: {image_weight}  |  Text weight: {text_weight}")

    # Weighted sum
    combined = image_weight * image_embeddings + text_weight * text_embeddings

    # Re-normalise combined embedding to unit sphere
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div by zero
    combined = combined / norms
    print(f"  ✓ Combined embeddings shape: {combined.shape}")

    # ── Save to disk ──────────────────────────────────────────────────────────
    np.save(str(COMBINED_PATH), combined)
    np.save(str(IMAGE_PATH), image_embeddings)
    np.save(str(TEXT_PATH), text_embeddings)
    print(f"  ✓ Saved combined → {COMBINED_PATH}")

    # ── Build metadata mapping ────────────────────────────────────────────────
    metadata = {
        "model_id": embedder.model_id,
        "embedding_dim": EMBEDDING_DIM,
        "image_weight": image_weight,
        "text_weight": text_weight,
        "n_products": n,
        "id_to_index": {},
        "index_to_id": {},
        "products": {},
    }

    for idx, row in products_df.iterrows():
        pid = row["id"]
        i = int(idx)
        metadata["id_to_index"][pid] = i
        metadata["index_to_id"][str(i)] = pid
        metadata["products"][pid] = {
            "index": i,
            "name": str(row["name"]),
            "brand": str(row["brand"]),
            "price_inr": float(row["price_inr"]) if not np.isnan(row["price_inr"]) else 0.0,
            "rating": float(row["rating"]) if not np.isnan(row["rating"]) else 0.0,
            "gender": str(row["gender"]),
            "wear_type": str(row["wear_type"]),
            "category": str(row["category"]),
            "category_label": str(row["category_label"]),
            "occasion": str(row["occasion"]),
            "image_path": str(row["image_path"]) if row["image_path"] else "",
            "embed_text": str(row["embed_text"])[:200],
            "has_image": bool(row["has_image"]),
        }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved → {METADATA_PATH}")

    return combined, metadata


def load_cached_embeddings() -> Tuple[np.ndarray, Dict]:
    """Load pre-computed embeddings from disk (must exist)."""
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {COMBINED_PATH}. "
            "Run generate_product_embeddings() first."
        )
    combined = np.load(str(COMBINED_PATH))
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return combined, metadata


_GLOBAL_EMBEDDER = None

def get_embedder() -> FashionEmbedder:
    """Return a loaded FashionEmbedder instance (singleton)."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = FashionEmbedder()
        _GLOBAL_EMBEDDER.load()
    return _GLOBAL_EMBEDDER


# ══════════════════════════════════════════════════════════════════════════════
# Embedding quality diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def run_embedding_diagnostics(embeddings: np.ndarray, metadata: Dict) -> None:
    """
    Print diagnostic statistics on the embedding space:
    - Norm distribution (should be ~1.0 for normalised vectors)
    - Intra-class similarity (same category should be more similar)
    - Nearest neighbours for 3 sample items
    """
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n" + "─" * 50)
    print("EMBEDDING DIAGNOSTICS")
    print("─" * 50)

    # Norm stats
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nNorm stats (should be ~1.0 for all):")
    print(f"  Min: {norms.min():.4f}  Max: {norms.max():.4f}  Mean: {norms.mean():.4f}")

    # Similarity matrix sample
    sim = cosine_similarity(embeddings[:20], embeddings[:20])
    off_diag = sim[np.triu_indices(20, k=1)]
    print(f"\nPairwise cosine similarity (first 20 products):")
    print(f"  Mean: {off_diag.mean():.4f}  Std: {off_diag.std():.4f}")
    print(f"  Min:  {off_diag.min():.4f}  Max: {off_diag.max():.4f}")

    # Sample nearest neighbours
    id_to_idx = metadata["id_to_index"]
    idx_to_id = metadata["index_to_id"]
    products_meta = metadata["products"]

    print("\nSample nearest neighbours (top-3 by cosine similarity):")
    sample_indices = [0, 15, 35]
    for i in sample_indices:
        pid = idx_to_id.get(str(i))
        if not pid:
            continue
        prod = products_meta[pid]
        sims = cosine_similarity([embeddings[i]], embeddings)[0]
        top_k = np.argsort(-sims)[1:4]  # skip self

        print(f"\n  Query: [{prod['category_label']}] {prod['name'][:45]}")
        for rank, j in enumerate(top_k, 1):
            npid = idx_to_id.get(str(j))
            if npid:
                np_prod = products_meta[npid]
                print(f"    {rank}. [{np_prod['category_label']}] "
                      f"{np_prod['name'][:40]} — sim={sims[j]:.3f}")


if __name__ == "__main__":
    from src.data.loader import load_products

    print("=" * 60)
    print("  PHASE 2: EMBEDDING GENERATION")
    print("=" * 60)

    products = load_products()
    embeddings, metadata = generate_product_embeddings(products, force_regenerate=False)
    run_embedding_diagnostics(embeddings, metadata)

    print("\n✓ Phase 2 complete — embeddings ready for FAISS indexing")
