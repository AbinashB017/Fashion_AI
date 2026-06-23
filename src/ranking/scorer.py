"""
src/ranking/scorer.py
----------------------
Phase 6: Multi-factor Compatibility Scoring Engine.

Every outfit recommendation gets a transparent, breakdown-able score:

  Final Score = 0.35 × Graph Score
              + 0.25 × Visual Score
              + 0.20 × Occasion Match
              + 0.10 × Category Compatibility
              + 0.10 × Style Match

Scores are in [0, 1] and displayed as percentages in the UI.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Category compatibility matrix
# Fashion rules encoded as expert knowledge
# ══════════════════════════════════════════════════════════════════════════════

# Score [0..1] for pairing category A with category B
# Based on standard fashion rules
CATEGORY_COMPAT = {
    # Tops + Bottoms (always compatible)
    ("formal-shirts", "trousers"): 1.0,
    ("formal-shirts", "jeans"): 0.75,
    ("casual-shirts", "jeans"): 1.0,
    ("casual-shirts", "chinos"): 1.0,
    ("casual-shirts", "trousers"): 0.8,
    ("linen-shirts", "chinos"): 1.0,
    ("linen-shirts", "shorts"): 0.9,
    ("party-shirts", "trousers"): 0.9,
    ("tshirts", "jeans"): 1.0,
    ("tshirts", "chinos"): 0.9,
    ("tshirts", "track-pants"): 0.9,
    ("polo-tshirts", "jeans"): 0.95,
    ("polo-tshirts", "chinos"): 1.0,
    ("sweatshirts", "track-pants"): 1.0,
    ("sweatshirts", "jeans"): 0.85,
    ("tops", "jeans"): 0.9,
    ("tops", "skirts"): 1.0,
    ("tops", "trousers"): 0.85,

    # Tops + Footwear
    ("formal-shirts", "formal-shoes"): 1.0,
    ("formal-shirts", "loafers"): 0.9,
    ("casual-shirts", "sneakers"): 0.9,
    ("casual-shirts", "loafers"): 0.85,
    ("tshirts", "sneakers"): 1.0,
    ("tshirts", "running-shoes"): 0.95,
    ("polo-tshirts", "sneakers"): 0.9,
    ("polo-tshirts", "loafers"): 0.85,

    # Women's
    ("party-dresses", "heels"): 1.0,
    ("party-dresses", "clutches"): 1.0,
    ("casual-dresses", "sneakers"): 0.9,
    ("casual-dresses", "flats"): 0.9,
    ("maxi-dresses", "flats"): 1.0,
    ("maxi-dresses", "sandals"): 0.95,
    ("co-ord-sets", "heels"): 0.9,
    ("co-ord-sets", "flats"): 0.85,
    ("formal-shirts", "heels"): 0.8,
    ("skirts", "heels"): 0.9,
    ("skirts", "sneakers"): 0.85,

    # Layers
    ("blazers", "formal-shirts"): 1.0,
    ("blazers", "trousers"): 0.95,
    ("denim-jackets", "tshirts"): 1.0,
    ("denim-jackets", "jeans"): 0.7,  # double denim needs care
    ("nehru-jackets", "kurta-sets"): 1.0,
    ("long-coats", "sweaters"): 0.9,
    ("long-coats", "jeans"): 0.85,
    ("suits", "formal-shirts"): 1.0,
    ("suits", "formal-shoes"): 1.0,

    # Ethnic
    ("kurta-sets", "ethnic-footwear"): 1.0,
    ("sherwanis", "ethnic-footwear"): 1.0,
    ("wedding-sarees", "ethnic-footwear"): 1.0,
    ("sharara-sets", "ethnic-footwear"): 1.0,
    ("salwar-suits", "ethnic-footwear"): 1.0,

    # Accessories
    ("party-dresses", "necklaces"): 0.9,
    ("formal-shirts", "watches"): 0.95,
    ("suits", "watches"): 1.0,
    ("casual-shirts", "caps"): 0.85,
    ("tshirts", "caps"): 0.9,
    ("maxi-dresses", "sunglasses"): 0.9,
    ("casual-dresses", "sunglasses"): 0.85,
    ("formal-shirts", "handbags"): 0.85,
    ("party-dresses", "clutches"): 1.0,
}

# Build reverse pairs too (symmetric)
_COMPAT_SYMMETRIC = {}
for (a, b), score in CATEGORY_COMPAT.items():
    _COMPAT_SYMMETRIC[(a, b)] = score
    _COMPAT_SYMMETRIC[(b, a)] = score

# Occasion compatibility — outfits shouldn't mix very different occasions
OCCASION_COMPAT = {
    ("office", "office"): 1.0,
    ("party", "party"): 1.0,
    ("casual", "casual"): 1.0,
    ("wedding", "wedding"): 1.0,
    ("festive", "festive"): 1.0,
    ("sports", "sports"): 1.0,
    ("vacation", "vacation"): 1.0,
    ("winter", "winter"): 1.0,
    # Cross-occasion pairs
    ("office", "casual"): 0.5,
    ("casual", "office"): 0.5,
    ("party", "casual"): 0.6,
    ("casual", "party"): 0.6,
    ("festive", "wedding"): 0.7,
    ("wedding", "festive"): 0.7,
    ("casual", "vacation"): 0.8,
    ("vacation", "casual"): 0.8,
    ("winter", "casual"): 0.6,
    ("casual", "winter"): 0.6,
}


# ══════════════════════════════════════════════════════════════════════════════
# Scoring engine
# ══════════════════════════════════════════════════════════════════════════════

class CompatibilityScorer:
    """
    Produces a multi-dimensional compatibility score for outfit items.
    All sub-scores are in [0, 1]; final score is a weighted average.
    """

    WEIGHTS = {
        "graph":      0.35,
        "visual":     0.25,
        "occasion":   0.20,
        "category":   0.10,
        "style":      0.10,
    }

    def __init__(self, graph=None, embeddings: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None):
        """
        graph     : FashionCompatibilityGraph instance
        embeddings: (N, 512) product embeddings array
        metadata  : product metadata dict with id_to_index
        """
        self.graph      = graph
        self.embeddings = embeddings
        self.metadata   = metadata

    # ── Pairwise scoring ──────────────────────────────────────────────────────

    def score_pair(self, id_a: str, prod_a: Dict,
                   id_b: str, prod_b: Dict) -> Dict:
        """
        Full scoring breakdown for a pair of items.
        Returns dict with all sub-scores + weighted final.
        """
        scores = {}

        # 1. Graph score
        scores["graph"] = self._graph_score(id_a, id_b)

        # 2. Visual (embedding cosine similarity)
        scores["visual"] = self._visual_score(id_a, id_b)

        # 3. Occasion match
        scores["occasion"] = self._occasion_score(
            prod_a.get("occasion", ""), prod_b.get("occasion", "")
        )

        # 4. Category compatibility
        scores["category"] = self._category_score(
            prod_a.get("category", ""), prod_b.get("category", "")
        )

        # 5. Style match (wear_type alignment)
        scores["style"] = self._style_score(
            prod_a.get("wear_type", ""), prod_b.get("wear_type", ""),
            prod_a.get("gender", ""), prod_b.get("gender", "")
        )

        # Weighted final
        final = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        scores["final"] = round(final, 4)
        scores["percentage"] = int(final * 100)

        return scores

    # ── Outfit-level scoring ──────────────────────────────────────────────────

    def score_outfit(self, outfit_items: List[Dict]) -> Dict:
        """
        Score a complete outfit (list of product dicts with 'id' field).
        Returns outfit-level breakdown averaged across all pairs.
        """
        if len(outfit_items) < 2:
            return {"final": 0.0, "percentage": 0, "pairs": []}

        pair_scores = []
        for item_a, item_b in combinations(outfit_items, 2):
            s = self.score_pair(
                item_a["id"], item_a,
                item_b["id"], item_b,
            )
            pair_scores.append(s)

        # Average across all pairs
        agg = {k: float(np.mean([ps[k] for ps in pair_scores]))
               for k in ["graph", "visual", "occasion", "category", "style", "final"]}
        agg["percentage"] = int(agg["final"] * 100)
        agg["pair_count"] = len(pair_scores)

        # Individual pair breakdown for UI display
        agg["pairs"] = pair_scores

        return agg

    def rank_outfits(self, outfits: List[List[Dict]]) -> List[Dict]:
        """
        Score and rank multiple outfit candidates.
        Returns list of dicts with outfit items + score, sorted best first.
        """
        ranked = []
        for outfit in outfits:
            score = self.score_outfit(outfit)
            ranked.append({"items": outfit, "score": score})

        ranked.sort(key=lambda x: x["score"]["final"], reverse=True)

        # Add rank
        for i, o in enumerate(ranked):
            o["rank"] = i + 1

        return ranked

    # ── Sub-score components ──────────────────────────────────────────────────

    def _graph_score(self, id_a: str, id_b: str) -> float:
        if self.graph is None:
            return 0.5  # neutral when graph unavailable
        return self.graph.compatibility_score(id_a, id_b)

    def _visual_score(self, id_a: str, id_b: str) -> float:
        """
        Cosine similarity between embeddings, mapped to [0, 1].

        Note: For fashion, MODERATE similarity is ideal (~0.5-0.7).
        Identical similarity (1.0) means same-type items.
        Very low similarity (<0.2) may indicate clashing aesthetics.

        We apply a bell-curve mapping: peak at ~0.55 cosine sim.
        """
        if self.embeddings is None or self.metadata is None:
            return 0.5

        id_to_idx = self.metadata.get("id_to_index", {})
        idx_a = id_to_idx.get(id_a)
        idx_b = id_to_idx.get(id_b)

        if idx_a is None or idx_b is None:
            return 0.5

        vec_a = self.embeddings[idx_a]
        vec_b = self.embeddings[idx_b]

        raw_sim = float(np.dot(vec_a, vec_b))  # already unit-normed
        raw_sim = max(-1.0, min(1.0, raw_sim))

        # Map: cosine in [-1, 1] → [0, 1] using moderate-peak transformation
        # Ideal fashion pairing: similar aesthetic but different category
        # Target cosine range ~[0.3, 0.7] → maps to high visual score
        visual = (raw_sim + 1.0) / 2.0  # linear to [0, 1]

        # Slight penalty for very high similarity (same-type items)
        if raw_sim > 0.85:
            visual *= 0.7

        return float(np.clip(visual, 0.0, 1.0))

    def _occasion_score(self, occ_a: str, occ_b: str) -> float:
        """Score occasion alignment between two items."""
        if not occ_a or not occ_b:
            return 0.5
        key = (occ_a.lower(), occ_b.lower())
        return OCCASION_COMPAT.get(key, 0.3)

    def _category_score(self, cat_a: str, cat_b: str) -> float:
        """Score category compatibility using the expert rules matrix."""
        if not cat_a or not cat_b:
            return 0.5
        key = (cat_a.lower(), cat_b.lower())
        score = _COMPAT_SYMMETRIC.get(key, None)
        if score is not None:
            return score
        # Same category = penalty (don't wear two shirts)
        if cat_a == cat_b:
            return 0.1
        return 0.4  # unknown pairing — neutral-ish

    def _style_score(self, wear_a: str, wear_b: str,
                     gender_a: str, gender_b: str) -> float:
        """
        Style alignment: same wear_type (both western or both ethnic) preferred.
        Gender mismatch is a hard penalty.
        """
        score = 0.5

        # Wear type alignment
        if wear_a and wear_b:
            # Allow footwear and accessories with any wear type
            neutral = {"footwear", "accessory"}
            if wear_a in neutral or wear_b in neutral:
                score = 0.8
            elif wear_a == wear_b:
                score = 1.0
            else:
                score = 0.2  # mixing western + ethnic is usually intentional fusion

        # Gender alignment
        if gender_a and gender_b:
            if gender_a != gender_b:
                score *= 0.5  # significant penalty for gender mismatch

        return float(np.clip(score, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# Utility: Format score for display
# ══════════════════════════════════════════════════════════════════════════════

def format_score_breakdown(score: Dict) -> str:
    """Format a compatibility score dict into a readable string."""
    lines = [
        f"Overall Compatibility: {score.get('percentage', 0)}%",
        f"  Graph Score     : {score.get('graph', 0)*100:.0f}%  (outfit co-occurrence)",
        f"  Visual Score    : {score.get('visual', 0)*100:.0f}%  (aesthetic similarity)",
        f"  Occasion Match  : {score.get('occasion', 0)*100:.0f}%  (context alignment)",
        f"  Category Match  : {score.get('category', 0)*100:.0f}%  (fashion rules)",
        f"  Style Match     : {score.get('style', 0)*100:.0f}%  (wear type / gender)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test without full pipeline
    scorer = CompatibilityScorer()

    # Test category scores
    test_pairs = [
        ("formal-shirts", "trousers"),
        ("party-dresses", "heels"),
        ("tshirts", "jeans"),
        ("formal-shirts", "formal-shirts"),  # same category — should penalise
        ("blazers", "formal-shirts"),
    ]
    print("Category compatibility scores:")
    for a, b in test_pairs:
        s = scorer._category_score(a, b)
        print(f"  {a:25} + {b:25} = {s:.2f}")
