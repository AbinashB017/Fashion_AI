"""
src/engine/outfit_engine.py
----------------------------
Phase 5 + 7: Hybrid Retrieval Engine + Outfit Candidate Generator.

This is the central orchestrator that brings together:
  - Intent extraction (Gemini)
  - Metadata filtering
  - FAISS vector search
  - Compatibility graph
  - Compatibility scoring
  - Fashion RAG
  - Outfit assembly

Produces 3 complete outfit candidates:
  Option 1: Classic (graph-first, highest compatibility)
  Option 2: Modern (vector-first, aesthetic variety)
  Option 3: Premium (price-optimised, elevated picks)
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Category slot mapping — which categories fill which outfit slots
SLOT_CATEGORIES = {
    "topwear": [
        "formal-shirts", "casual-shirts", "party-shirts", "linen-shirts",
        "tshirts", "polo-tshirts", "sweatshirts", "tops", "activewear",
        "sweaters",
    ],
    "bottomwear": [
        "trousers", "jeans", "chinos", "track-pants", "shorts",
        "skirts", "leggings",
    ],
    "fullbody": [
        "party-dresses", "casual-dresses", "maxi-dresses",
        "co-ord-sets", "suits", "sherwanis", "wedding-sarees",
        "sharara-sets", "salwar-suits", "kurta-sets",
    ],
    "layer": [
        "blazers", "denim-jackets", "long-coats", "nehru-jackets",
    ],
    "footwear": [
        "heels", "boots", "sneakers", "loafers", "formal-shoes",
        "sandals", "flats", "ethnic-footwear", "running-shoes",
    ],
    "accessory": [
        "handbags", "clutches", "necklaces", "earrings",
        "watches", "caps", "sunglasses",
    ],
}

# Reverse lookup: category → slot
CAT_TO_SLOT = {}
for slot, cats in SLOT_CATEGORIES.items():
    for cat in cats:
        CAT_TO_SLOT[cat] = slot

# Occasion → preferred categories for each slot
OCCASION_SLOT_PREFS = {
    "office": {
        "topwear": ["formal-shirts", "tops"],
        "bottomwear": ["trousers", "jeans"],
        "layer": ["blazers"],
        "footwear": ["formal-shoes", "heels", "loafers"],
        "accessory": ["handbags", "watches"],
    },
    "party": {
        "topwear": ["party-shirts", "tops"],
        "bottomwear": ["jeans", "trousers"],
        "fullbody": ["party-dresses", "co-ord-sets"],
        "footwear": ["heels", "loafers", "boots"],
        "accessory": ["clutches", "necklaces", "earrings"],
    },
    "casual": {
        "topwear": ["casual-shirts", "tshirts", "polo-tshirts", "tops"],
        "bottomwear": ["jeans", "chinos", "shorts", "skirts"],
        "layer": ["denim-jackets"],
        "footwear": ["sneakers", "loafers", "flats", "sandals"],
        "accessory": ["handbags", "caps", "sunglasses"],
    },
    "wedding": {
        "fullbody": ["sherwanis", "wedding-sarees", "suits"],
        "topwear": ["formal-shirts"],
        "footwear": ["ethnic-footwear", "formal-shoes", "heels"],
        "accessory": ["necklaces", "watches"],
    },
    "festive": {
        "fullbody": ["kurta-sets", "sharara-sets", "salwar-suits"],
        "topwear": ["formal-shirts"],
        "layer": ["nehru-jackets"],
        "footwear": ["ethnic-footwear"],
        "accessory": ["necklaces", "earrings"],
    },
    "sports": {
        "topwear": ["tshirts", "sweatshirts", "activewear"],
        "bottomwear": ["track-pants", "shorts", "leggings"],
        "footwear": ["running-shoes", "sneakers"],
        "accessory": ["caps"],
    },
    "vacation": {
        "topwear": ["linen-shirts", "tshirts", "tops"],
        "bottomwear": ["shorts", "jeans", "chinos"],
        "fullbody": ["maxi-dresses", "casual-dresses"],
        "footwear": ["sandals", "sneakers", "flats"],
        "accessory": ["sunglasses", "handbags"],
    },
    "winter": {
        "topwear": ["sweaters", "sweatshirts"],
        "bottomwear": ["jeans", "trousers"],
        "layer": ["long-coats", "denim-jackets"],
        "footwear": ["boots"],
        "accessory": ["handbags"],
    },
}


class OutfitEngine:
    """
    Central orchestration engine for outfit generation.

    Takes a user query, extracts intent, retrieves candidates,
    scores them, and returns 3 complete outfit options with
    compatibility scores and explanations.
    """

    def __init__(self, products_df, outfits_df,
                 faiss_index=None, graph=None,
                 scorer=None, rag=None):
        self.products_df = products_df
        self.outfits_df  = outfits_df
        self.faiss_index = faiss_index
        self.graph       = graph
        self.scorer      = scorer
        self.rag         = rag

        # Build quick lookups
        self._id_to_product = {
            row["id"]: row.to_dict()
            for _, row in products_df.iterrows()
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════════════════════

    def generate_outfits(self, user_query: str,
                          intent: Optional[Dict] = None,
                          n_outfits: int = 3) -> Dict:
        """
        Full pipeline: query → 3 outfit candidates with scores + explanations.

        Returns:
        {
          "intent": {...},
          "outfits": [
            {
              "rank": 1,
              "theme": "Classic Office",
              "items": [...],
              "score": {"final": 0.82, "percentage": 82, ...},
              "explanation": "...",
              "rag_sources": [...]
            },
            ...
          ],
          "total_candidates": N
        }
        """
        from src.llm.gemini_client import extract_intent, generate_explanation

        # ── Step 1: Intent extraction ─────────────────────────────────────────
        if intent is None:
            intent = extract_intent(user_query)
        print(f"[Engine] Intent: {intent}")

        # ── Step 2: Hybrid retrieval ──────────────────────────────────────────
        candidate_pool = self._hybrid_retrieve(intent, user_query)
        print(f"[Engine] Retrieved {len(candidate_pool)} candidates")

        # ── Step 3: Assemble outfit candidates ────────────────────────────────
        outfit_candidates = self._assemble_outfits(candidate_pool, intent, n_outfits)
        print(f"[Engine] Assembled {len(outfit_candidates)} outfit candidates")

        if not outfit_candidates:
            return {"intent": intent, "outfits": [], "error": "No compatible outfits found"}

        # ── Step 4: Score + rank ──────────────────────────────────────────────
        if self.scorer:
            ranked = self.scorer.rank_outfits(outfit_candidates)
        else:
            ranked = [{"items": o, "score": {"final": 0.5, "percentage": 50}, "rank": i+1}
                      for i, o in enumerate(outfit_candidates)]

        # ── Step 5: Name themes ───────────────────────────────────────────────
        themes = self._assign_themes(ranked, intent)

        # ── Step 6: RAG + explanations ────────────────────────────────────────
        import concurrent.futures
        final_outfits = [None] * len(ranked)
        
        def process_outfit(i, outfit_data, theme):
            # Retrieve relevant stylist rationales
            rag_query = f"{user_query} {intent.get('occasion','')} {intent.get('gender','')}"
            rag_sources = []
            rag_context = ""
            if self.rag:
                try:
                    rag_sources = self.rag.retrieve(
                        rag_query, k=2,
                        occasion_filter=intent.get("occasion"),
                        gender_filter=intent.get("gender"),
                    )
                    rag_context = self.rag.format_context_for_llm(rag_sources)
                except Exception as e:
                    print(f"[Engine] RAG retrieval failed: {e}")

            # Generate explanation
            outfit_for_explain = {
                "items": outfit_data["items"],
                "occasion": intent.get("occasion", ""),
                "wear_type": intent.get("wear_type", "western"),
                "gender": intent.get("gender", ""),
                "palette": self._infer_palette(outfit_data["items"]),
            }
            try:
                explanation = generate_explanation(
                    user_query=user_query,
                    outfit=outfit_for_explain,
                    rag_context=rag_context,
                    outfit_number=i + 1,
                    outfit_theme=theme,
                    compatibility_score=outfit_data["score"]["percentage"],
                )
            except Exception as e:
                print(f"[Engine] Explanation generation failed: {e}")
                explanation = self._fallback_explanation(outfit_data["items"], theme)

            return {
                "rank": i + 1,
                "theme": theme,
                "items": outfit_data["items"],
                "score": outfit_data["score"],
                "explanation": explanation,
                "rag_sources": [
                    {"rationale": r["rationale"], "theme": r["theme"]}
                    for r in rag_sources
                ],
                "palette": self._infer_palette(outfit_data["items"]),
                "total_price": sum(
                    item.get("price_inr", 0) for item in outfit_data["items"]
                    if item.get("price_inr")
                ),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_outfits) as executor:
            futures = {executor.submit(process_outfit, i, o, t): i 
                       for i, (o, t) in enumerate(zip(ranked, themes))}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                final_outfits[idx] = future.result()

        return {
            "intent": intent,
            "outfits": final_outfits,
            "total_candidates": len(outfit_candidates),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Hybrid retrieval
    # ══════════════════════════════════════════════════════════════════════════

    def _hybrid_retrieve(self, intent: Dict, query: str) -> Dict[str, List[Dict]]:
        """
        Retrieve product candidates per slot using 3 signals:
          1. Metadata filter (gender, occasion, wear_type)
          2. FAISS semantic search
          3. Compatibility graph edges

        Returns: {slot_name: [product_dicts]}
        """
        gender    = intent.get("gender")
        
        if not gender:
            q_lower = query.lower()
            if any(w in q_lower for w in ["women", "female", "woman", "girl", "she", "ladies"]):
                gender = "women"
            elif any(w in q_lower for w in [" men ", " man ", "male", "guy", " he ", "boys", "groom"]):
                gender = "men"
            elif q_lower.startswith("men ") or q_lower.endswith(" men") or q_lower == "men":
                gender = "men"
            if q_lower.startswith("man ") or q_lower.endswith(" man") or q_lower == "man":
                gender = "men"
                
            if gender:
                intent["gender"] = gender

        occasion  = intent.get("occasion")
        wear_type = intent.get("wear_type")
        style     = intent.get("style")
        age       = intent.get("age")
        user_profile = intent.get("user_profile", {})

        # Filters for FAISS
        faiss_filters = {
            k: v for k, v in {
                "gender": gender,
                "occasion": occasion,
                "wear_type": wear_type,
            }.items() if v
        }

        # Determine required slots for this occasion
        slot_prefs = OCCASION_SLOT_PREFS.get(occasion or "casual", {})

        # ── 1. Start with metadata-filtered products per slot ─────────────────
        slot_candidates = defaultdict(list)
        for _, prod in self.products_df.iterrows():
            p = prod.to_dict()

            # Gender filter
            if gender and p.get("gender") not in (gender, "unisex"):
                continue

            # Wear type filter (ethnic/western)
            prod_cat = p.get("category", "")
            slot = CAT_TO_SLOT.get(prod_cat, "unknown")
            if slot == "unknown":
                continue

            # Occasion-based category preference
            slot_cats = slot_prefs.get(slot, [])
            pref_score = 1.0 if prod_cat in slot_cats else 0.5

            p["pref_score"] = pref_score
            p["slot"] = slot
            slot_candidates[slot].append(p)

        # ── 2. FAISS semantic boost with Query Rewriting ───────────────────────
        if self.faiss_index:
            try:
                # Semantic Query Rewriting
                semantic_parts = []
                if style: semantic_parts.append(style)
                if wear_type: semantic_parts.append(wear_type)
                semantic_parts.append("outfit for")
                if occasion: semantic_parts.append(occasion)
                if age: semantic_parts.append(f"({age})")
                if gender: semantic_parts.append(f"({gender})")
                
                enriched_query = " ".join(semantic_parts) if len(semantic_parts) > 1 else query
                # Fallback to original query if intent is too sparse
                if not style and not occasion and not age:
                    enriched_query = query
                    
                faiss_results = self.faiss_index.text_search(
                    enriched_query, k=20, filters=faiss_filters
                )
                
                top_faiss_ids = []
                for idx, r in enumerate(faiss_results):
                    pid = r["id"]
                    if idx < 5:
                        top_faiss_ids.append(pid)
                        
                    cat = r.get("category", "")
                    slot = CAT_TO_SLOT.get(cat, "unknown")
                    if slot == "unknown":
                        continue
                        
                    # Find and boost this product's score
                    for cand in slot_candidates[slot]:
                        if cand["id"] == pid:
                            cand["pref_score"] = min(
                                cand.get("pref_score", 0.5) + r["score"] * 0.4,
                                1.0
                            )
                            cand["faiss_score"] = r["score"]
                            break
                            
                # ── 2.5 Graph Expansion ───────────────────────────────────────
                # Inject highly compatible neighbors of the top 5 FAISS results
                if self.graph:
                    for pid in top_faiss_ids:
                        neighbors = self.graph.get_compatible_items(pid, top_k=5)
                        for neighbor_id in neighbors:
                            # Boost neighbor score in candidate pool
                            for s in slot_candidates:
                                for cand in slot_candidates[s]:
                                    if cand["id"] == neighbor_id:
                                        cand["pref_score"] = min(cand.get("pref_score", 0.5) + 0.3, 1.0)
                                        cand["graph_boost"] = True
                                        break
                                        
            except Exception as e:
                print(f"[Engine] FAISS retrieval error: {e}")

        # ── 3. Sort each slot by score ─────────────────────────────────────────
        for slot in slot_candidates:
            slot_candidates[slot].sort(
                key=lambda x: (x.get("pref_score", 0), x.get("rating", 0)),
                reverse=True
            )

        return dict(slot_candidates)

    # ══════════════════════════════════════════════════════════════════════════
    # Outfit assembly
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_diversity_penalty(self, pool: Dict[str, List[Dict]], outfit: List[Dict]):
        if not outfit:
            return
        used_ids = {item["id"] for item in outfit}
        for slot, candidates in pool.items():
            for cand in candidates:
                if cand["id"] in used_ids:
                    cand["pref_score"] = cand.get("pref_score", 0.5) * 0.1
                    cand["faiss_score"] = cand.get("faiss_score", 0.0) * 0.1

    def _assemble_outfits(self, candidate_pool: Dict[str, List],
                           intent: Dict, n_outfits: int) -> List[List[Dict]]:
        """
        Assemble n_outfits complete outfit lists from the candidate pool.

        Strategy:
          - Outfit 1 (Classic): Top-ranked item per slot
          - Outfit 2 (Modern): 2nd pick for hero/key item, fresh combinations
          - Outfit 3 (Premium): Highest-price items per slot
        """
        outfits = []
        occasion  = intent.get("occasion", "casual")
        wear_type = intent.get("wear_type", "western")
        gender    = intent.get("gender")
        user_profile = intent.get("user_profile", {})
        budget = intent.get("budget") or user_profile.get("budget")
        color_pref = intent.get("color_preference") or user_profile.get("preferred_colors", [])

        # Apply profile/intent re-ranking boosts to candidate_pool
        for slot in candidate_pool:
            for cand in candidate_pool[slot]:
                # Price logic
                price = cand.get("price_inr", 0)
                if budget == "low" and price > 0 and price < 1500:
                    cand["pref_score"] = min(cand.get("pref_score", 0.5) + 0.15, 1.0)
                elif budget == "high" and price > 3000:
                    cand["pref_score"] = min(cand.get("pref_score", 0.5) + 0.15, 1.0)
                
                # Color logic
                if color_pref:
                    if isinstance(color_pref, str): color_pref = [color_pref]
                    text = str(cand.get("base_colour", "")) + " " + str(cand.get("name", ""))
                    if any(c.lower() in text.lower() for c in color_pref):
                        cand["pref_score"] = min(cand.get("pref_score", 0.5) + 0.2, 1.0)

        # Determine if it's ethnic or western to pick correct slot schema
        is_ethnic = wear_type == "ethnic" or occasion in ("wedding", "festive")

        def _pick(pool, offset=0):
            if pool and offset < len(pool):
                return pool[offset]
            return None

        # ── Outfit 1: Classic ─────────────────────────────────────────────────
        outfit1 = self._build_outfit(
            candidate_pool, occasion, is_ethnic, gender,
            strategy="classic", offset=0
        )
        if outfit1:
            outfits.append(outfit1)
            self._apply_diversity_penalty(candidate_pool, outfit1)

        # ── Outfit 2: Modern (different hero/main item) ───────────────────────
        outfit2 = self._build_outfit(
            candidate_pool, occasion, is_ethnic, gender,
            strategy="modern", offset=1
        )
        if outfit2:
            outfits.append(outfit2)
            self._apply_diversity_penalty(candidate_pool, outfit2)

        # ── Outfit 3: Premium (higher price picks) ────────────────────────────
        outfit3 = self._build_outfit(
            candidate_pool, occasion, is_ethnic, gender,
            strategy="premium", offset=2
        )
        if outfit3:
            outfits.append(outfit3)

        # Deduplicate by full item-set fingerprint (not just hero)
        seen_sets = []
        unique = []
        for o in outfits:
            ids = frozenset(x["id"] for x in o)
            if ids not in seen_sets:
                seen_sets.append(ids)
                unique.append(o)

        return unique[:n_outfits]

    def _build_outfit(self, pool: Dict, occasion: str, is_ethnic: bool,
                       gender: Optional[str], strategy: str, offset: int) -> List[Dict]:
        """
        Build one complete outfit from the candidate pool.
        """
        outfit = []
        outfit_gender = gender

        if is_ethnic:
            # Ethnic outfit: fullbody → footwear → accessory
            slots_to_fill = ["fullbody", "footwear", "accessory"]
        else:
            # Western outfit schema
            fullbody = pool.get("fullbody", [])
            if fullbody and occasion in ("party", "vacation") and outfit_gender == "women":
                # Women's party/vacation: use dress as fullbody
                slots_to_fill = ["fullbody", "footwear", "accessory"]
            else:
                slots_to_fill = ["topwear", "bottomwear", "footwear", "accessory"]
                # Add layer for office/winter
                if occasion in ("office", "winter"):
                    slots_to_fill.insert(2, "layer")

        for slot in slots_to_fill:
            candidates = pool.get(slot, [])
            if not candidates:
                continue

            # Filter candidates by outfit_gender if it's set dynamically
            filtered_candidates = []
            for c in candidates:
                cg = c.get("gender")
                if outfit_gender:
                    if cg in (outfit_gender, "unisex", None):
                        filtered_candidates.append(c)
                else:
                    filtered_candidates.append(c)
                    
            if not filtered_candidates:
                continue

            # Sort by strategy
            if strategy == "premium":
                candidates_sorted = sorted(filtered_candidates,
                    key=lambda x: x.get("price_inr", 0), reverse=True)
            elif strategy == "modern":
                candidates_sorted = sorted(filtered_candidates,
                    key=lambda x: (x.get("faiss_score", 0), x.get("pref_score", 0)),
                    reverse=True)
            else:  # classic
                candidates_sorted = sorted(filtered_candidates,
                    key=lambda x: x.get("pref_score", 0), reverse=True)

            # Use offset for variety
            pick_idx = offset if slot in ("topwear", "fullbody") else 0
            pick = candidates_sorted[min(pick_idx, len(candidates_sorted) - 1)]

            # Lock the gender for the rest of the outfit if we didn't have one
            if not outfit_gender and pick.get("gender") not in ("unisex", None, ""):
                outfit_gender = pick.get("gender")

            # Avoid duplicates (same item in multiple slots)
            if not any(o["id"] == pick["id"] for o in outfit):
                outfit.append(pick)

        return outfit if len(outfit) >= 2 else []

    # ══════════════════════════════════════════════════════════════════════════
    # Helper methods
    # ══════════════════════════════════════════════════════════════════════════

    def _assign_themes(self, ranked_outfits: List[Dict],
                        intent: Dict) -> List[str]:
        """Assign descriptive themes to each outfit option."""
        occasion = intent.get("occasion", "casual")
        wear_type = intent.get("wear_type", "western")
        gender = intent.get("gender", "")

        theme_map = {
            "office":   ["Classic Office", "Modern Professional", "Premium Executive"],
            "party":    ["Evening Statement", "Night Out Chic", "Party Glamour"],
            "casual":   ["Weekend Casual", "Street Style", "Everyday Edit"],
            "wedding":  ["Ceremony Ready", "Reception Elegant", "Celebration"],
            "festive":  ["Festive Traditional", "Ethnic Chic", "Celebration Look"],
            "sports":   ["Active Athleisure", "Workout Ready", "Sport Mode"],
            "vacation": ["Holiday Breeze", "Travel-Ready", "Resort Casual"],
            "winter":   ["Winter Layers", "Cold Day Chic", "Cozy Elevated"],
        }

        themes = theme_map.get(occasion, ["Look 1", "Look 2", "Look 3"])
        return themes[:len(ranked_outfits)]

    def _infer_palette(self, items: List[Dict]) -> str:
        """Infer a colour palette string from item names/descriptions."""
        colors = set()
        color_keywords = ["black", "white", "navy", "blue", "red", "green",
                          "brown", "grey", "cream", "gold", "pink", "maroon",
                          "olive", "beige", "purple", "burgundy"]
        for item in items:
            text = (item.get("name", "") + " " + item.get("tags", "")).lower()
            for c in color_keywords:
                if c in text:
                    colors.add(c)
        return " / ".join(list(colors)[:4]) if colors else "mixed"

    def _fallback_explanation(self, items: List[Dict], theme: str) -> str:
        names = [i.get("name", "item")[:25] for i in items[:3]]
        return (
            f"This {theme} look brings together {' and '.join(names)} "
            f"for a cohesive, well-balanced ensemble. "
            f"The combination is thoughtfully curated to complement the occasion "
            f"while keeping the overall look polished and put-together."
        )

    # ── Image upload mode ─────────────────────────────────────────────────────

    def find_complements_for_image(self, image_path: str,
                                    intent: Optional[Dict] = None) -> Dict:
        """
        Bonus feature: Upload an image → find complementary items.
        Uses image embedding to find visually compatible products.
        """
        if not self.faiss_index:
            return {"error": "FAISS index not available"}

        try:
            # Find visually similar items
            similar = self.faiss_index.image_search(image_path, k=15)
            if not similar:
                return {"error": "Could not process image"}

            # The uploaded item's category (inferred from top match)
            top_match = similar[0]
            source_cat = top_match.get("category", "")
            source_slot = CAT_TO_SLOT.get(source_cat, "unknown")

            # Find complements from different slots
            complements = defaultdict(list)
            for prod in similar[1:]:
                slot = CAT_TO_SLOT.get(prod.get("category", ""), "unknown")
                if slot != source_slot and slot != "unknown":
                    complements[slot].append(prod)

            return {
                "source_item": top_match,
                "source_slot": source_slot,
                "complements": dict(complements),
                "message": f"Found {sum(len(v) for v in complements.values())} complement items"
            }
        except Exception as e:
            return {"error": str(e)}
