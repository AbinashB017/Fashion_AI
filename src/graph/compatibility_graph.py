"""
src/graph/compatibility_graph.py
---------------------------------
Phase 4: Fashion Compatibility Graph using NetworkX.

Architecture:
  - Each of the 68 products is a NODE with full attribute metadata
  - Each outfit co-occurrence creates EDGES between items
  - Edge attributes: outfit_id, roles (hero→footwear etc.), weight (frequency)
  - The graph captures expert stylist knowledge as a structured network

Key capabilities:
  1. Graph-based compatibility score between any two products
  2. Neighbourhood retrieval ("what goes with this shirt?")
  3. Outfit-aware recommendations (leverage expert curations)
  4. Centrality analysis (identify "staple" items)
  5. Graph visualisation
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
from itertools import combinations

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

GRAPH_CACHE = ROOT / "embeddings" / "compatibility_graph.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# Compatibility Graph
# ══════════════════════════════════════════════════════════════════════════════

class FashionCompatibilityGraph:
    """
    A directed multigraph where:
      - Nodes  = fashion products (with metadata attributes)
      - Edges  = outfit co-occurrence relationships
      - Weight = number of expert outfits featuring both items together

    We use an undirected weighted graph (UndirectedGraph) since fashion
    compatibility is symmetric: if A pairs with B, B pairs with A.
    """

    SLOT_ROLES = {
        "hero": "hero",
        "second": "bottom/second",
        "layer": "layer",
        "footwear": "footwear",
        "accessory_1": "accessory",
        "accessory_2": "accessory",
    }

    def __init__(self):
        self.G = nx.Graph()
        self._built = False

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, products_df, outfits_df) -> None:
        """
        Construct the compatibility graph from products + outfits data.

        Node attributes:
          id, name, brand, gender, category, occasion, wear_type,
          price_inr, rating, image_path, platform, centrality

        Edge attributes:
          weight (co-occurrence count), outfits (list), role_pairs (list)
        """
        print("[Graph] Building fashion compatibility graph...")

        # ── Add all product nodes ─────────────────────────────────────────────
        for _, row in products_df.iterrows():
            self.G.add_node(
                row["id"],
                name=str(row["name"]),
                brand=str(row["brand"]),
                gender=str(row["gender"]),
                wear_type=str(row["wear_type"]),
                category=str(row["category"]),
                category_label=str(row["category_label"]),
                occasion=str(row["occasion"]),
                price_inr=float(row["price_inr"]) if not np.isnan(row["price_inr"]) else 0.0,
                rating=float(row["rating"]) if not np.isnan(row["rating"]) else 0.0,
                image_path=str(row["image_path"]) if row["image_path"] else "",
                platform=str(row["platform"]),
                tags=row.get("tags", ""),
            )

        # ── Add outfit co-occurrence edges ────────────────────────────────────
        id_cols_roles = {
            "hero_id":        "hero",
            "second_id":      "bottom/second",
            "layer_id":       "layer",
            "footwear_id":    "footwear",
            "accessory_1_id": "accessory",
            "accessory_2_id": "accessory",
        }

        for _, outfit in outfits_df.iterrows():
            outfit_id = outfit["outfit_id"]

            # Gather all (product_id, role) pairs in this outfit
            slot_items = []
            for col, role in id_cols_roles.items():
                val = outfit.get(col, "")
                if isinstance(val, str) and val.strip():
                    slot_items.append((val.strip(), role))

            # Create edges for every pair of items in this outfit
            for (id_a, role_a), (id_b, role_b) in combinations(slot_items, 2):
                if id_a not in self.G or id_b not in self.G:
                    continue  # skip if product not in our 68

                if self.G.has_edge(id_a, id_b):
                    # Strengthen existing edge
                    self.G[id_a][id_b]["weight"] += 1
                    self.G[id_a][id_b]["outfits"].append(outfit_id)
                    self.G[id_a][id_b]["role_pairs"].append((role_a, role_b))
                else:
                    # Create new edge
                    self.G.add_edge(
                        id_a, id_b,
                        weight=1,
                        outfits=[outfit_id],
                        role_pairs=[(role_a, role_b)],
                        occasion=str(outfit.get("occasion", "")),
                        gender=str(outfit.get("gender", "")),
                        theme=str(outfit.get("theme", "")),
                        palette=str(outfit.get("palette", "")),
                        stylist_rationale=str(outfit.get("stylist_rationale", "")),
                    )

        # ── Compute centrality metrics ─────────────────────────────────────────
        degree_centrality   = nx.degree_centrality(self.G)
        betweenness         = nx.betweenness_centrality(self.G, weight="weight")
        pagerank            = nx.pagerank(self.G, weight="weight")

        for node in self.G.nodes():
            self.G.nodes[node]["degree_centrality"]   = degree_centrality.get(node, 0.0)
            self.G.nodes[node]["betweenness"]          = betweenness.get(node, 0.0)
            self.G.nodes[node]["pagerank"]             = pagerank.get(node, 0.0)
            self.G.nodes[node]["degree"]               = self.G.degree(node)

        self._built = True

        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        print(f"[Graph] ✓ Graph built: {n_nodes} nodes, {n_edges} edges")
        print(f"[Graph]   Connected components: {nx.number_connected_components(self.G)}")

        # Top staple items by degree
        top_nodes = sorted(self.G.nodes(data=True),
                           key=lambda x: x[1].get("degree", 0), reverse=True)[:5]
        print(f"[Graph]   Top staple items:")
        for nid, ndata in top_nodes:
            print(f"    [{ndata['degree']}°] {ndata['name'][:45]}")

    # ── Compatibility scoring ─────────────────────────────────────────────────

    def compatibility_score(self, id_a: str, id_b: str) -> float:
        """
        Graph-based compatibility score between two products.

        Score components:
          1. Direct edge: items appear together in expert outfits (+high)
          2. Edge weight: more shared outfits = higher score
          3. Common neighbours: shared neighbours (transitivity)
          4. Occasion match: same occasion on nodes

        Returns a score in [0, 1].
        """
        if not self._built:
            return 0.0
        if id_a not in self.G or id_b not in self.G:
            return 0.0

        score = 0.0

        # Direct edge score
        if self.G.has_edge(id_a, id_b):
            weight = self.G[id_a][id_b]["weight"]
            # Normalise: max weight is ~4 (a staple item appears in 4 outfits)
            score += min(weight / 4.0, 1.0) * 0.7

        # Common neighbours (Jaccard similarity on adjacency)
        neighbors_a = set(self.G.neighbors(id_a))
        neighbors_b = set(self.G.neighbors(id_b))
        union = neighbors_a | neighbors_b
        if union:
            jaccard = len(neighbors_a & neighbors_b) / len(union)
            score += jaccard * 0.2

        # Occasion alignment bonus
        occ_a = self.G.nodes[id_a].get("occasion", "")
        occ_b = self.G.nodes[id_b].get("occasion", "")
        if occ_a and occ_b and occ_a == occ_b:
            score += 0.1

        return min(score, 1.0)

    def pairwise_outfit_score(self, product_ids: List[str]) -> float:
        """
        Average pairwise compatibility score for a complete outfit.
        Used to rank multiple outfit candidates.
        """
        if len(product_ids) < 2:
            return 0.0
        pairs = list(combinations(product_ids, 2))
        scores = [self.compatibility_score(a, b) for a, b in pairs]
        return float(np.mean(scores))

    # ── Neighbourhood retrieval ───────────────────────────────────────────────

    def get_compatible_items(self, product_id: str,
                             category_filter: Optional[List[str]] = None,
                             occasion_filter: Optional[str] = None,
                             gender_filter: Optional[str] = None,
                             top_k: int = 10) -> List[Dict]:
        """
        Return items that are directly connected to `product_id` in the graph,
        ranked by edge weight (co-occurrence frequency).

        This is the core of graph-based recommendation:
        "Given item X, what have expert stylists paired it with?"
        """
        if product_id not in self.G:
            return []

        neighbors = []
        for nbr in self.G.neighbors(product_id):
            edge = self.G[product_id][nbr]
            nbr_data = self.G.nodes[nbr]

            # Apply filters
            if category_filter:
                if (nbr_data.get("category") not in category_filter and
                        nbr_data.get("category_label") not in category_filter):
                    continue
            if occasion_filter:
                if nbr_data.get("occasion", "") != occasion_filter:
                    continue
            if gender_filter:
                g = nbr_data.get("gender", "")
                if g != gender_filter and g not in ("unisex", ""):
                    continue

            neighbors.append({
                "id": nbr,
                "name": nbr_data.get("name", ""),
                "category": nbr_data.get("category", ""),
                "category_label": nbr_data.get("category_label", ""),
                "gender": nbr_data.get("gender", ""),
                "occasion": nbr_data.get("occasion", ""),
                "price_inr": nbr_data.get("price_inr", 0.0),
                "rating": nbr_data.get("rating", 0.0),
                "image_path": nbr_data.get("image_path", ""),
                "graph_weight": edge.get("weight", 1),
                "shared_outfits": edge.get("outfits", []),
                "role_pairs": edge.get("role_pairs", []),
                "stylist_rationale": edge.get("stylist_rationale", ""),
            })

        # Sort by graph weight (higher = more expert co-occurrences)
        neighbors.sort(key=lambda x: x["graph_weight"], reverse=True)
        return neighbors[:top_k]

    def get_outfit_outgoing_items(self, hero_id: str,
                                   gender: str = None,
                                   occasion: str = None) -> Dict[str, List[Dict]]:
        """
        Given a hero item, return compatible items grouped by fashion slot.
        Returns: {"footwear": [...], "bottom/second": [...], "accessory": [...], ...}
        """
        if hero_id not in self.G:
            return {}

        slot_items = defaultdict(list)
        for nbr in self.G.neighbors(hero_id):
            edge = self.G[hero_id][nbr]
            nbr_data = self.G.nodes[nbr]

            # Gender filter
            if gender:
                nbr_gender = nbr_data.get("gender", "")
                if nbr_gender != gender and nbr_gender not in ("unisex", ""):
                    continue

            # Occasion filter
            if occasion:
                if nbr_data.get("occasion", "") != occasion:
                    # Still include but with lower weight
                    pass

            for role_a, role_b in edge.get("role_pairs", []):
                # Determine slot of neighbour
                slot = role_b if role_a == "hero" else role_a
                slot_items[slot].append({
                    "id": nbr,
                    "name": nbr_data.get("name", ""),
                    "category_label": nbr_data.get("category_label", ""),
                    "occasion": nbr_data.get("occasion", ""),
                    "image_path": nbr_data.get("image_path", ""),
                    "price_inr": nbr_data.get("price_inr", 0.0),
                    "graph_weight": edge["weight"],
                    "rationale": edge.get("stylist_rationale", ""),
                })

        # Deduplicate and sort per slot
        for slot in slot_items:
            seen = set()
            unique = []
            for item in sorted(slot_items[slot], key=lambda x: -x["graph_weight"]):
                if item["id"] not in seen:
                    seen.add(item["id"])
                    unique.append(item)
            slot_items[slot] = unique

        return dict(slot_items)

    # ── Staple items ──────────────────────────────────────────────────────────

    def get_staple_items(self, gender: Optional[str] = None,
                          occasion: Optional[str] = None,
                          top_k: int = 10) -> List[Dict]:
        """
        Return high-centrality items — these are the 'wardrobe staples'
        that pair well with many other items in the graph.
        """
        nodes = []
        for nid, ndata in self.G.nodes(data=True):
            if gender and ndata.get("gender", "") != gender:
                continue
            if occasion and ndata.get("occasion", "") != occasion:
                continue
            nodes.append({
                "id": nid,
                "name": ndata.get("name", ""),
                "category_label": ndata.get("category_label", ""),
                "degree": ndata.get("degree", 0),
                "pagerank": ndata.get("pagerank", 0.0),
                "betweenness": ndata.get("betweenness", 0.0),
                "image_path": ndata.get("image_path", ""),
            })

        nodes.sort(key=lambda x: (x["degree"], x["pagerank"]), reverse=True)
        return nodes[:top_k]

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path = GRAPH_CACHE) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.G, f)
        print(f"[Graph] ✓ Saved → {path}")

    def load(self, path: Path = GRAPH_CACHE) -> None:
        with open(path, "rb") as f:
            self.G = pickle.load(f)
        self._built = True
        print(f"[Graph] ✓ Loaded: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    # ── Visualisation ─────────────────────────────────────────────────────────

    def visualise(self, output_path: Optional[str] = None,
                  highlight_id: Optional[str] = None) -> None:
        """
        Draw the compatibility graph. Nodes coloured by category,
        edge thickness by weight.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        DARK = "#0F0F1A"
        CARD = "#1A1A2E"

        # Assign colours by category type
        category_types = {
            "formal-shirts": "#4F8EF7",
            "casual-shirts": "#4F8EF7",
            "party-shirts": "#4F8EF7",
            "linen-shirts": "#4F8EF7",
            "tshirts": "#00D4FF",
            "polo-tshirts": "#00D4FF",
            "sweatshirts": "#00D4FF",
            "tops": "#FF6B9D",
            "trousers": "#E94560",
            "jeans": "#E94560",
            "chinos": "#E94560",
            "track-pants": "#E94560",
            "shorts": "#E94560",
            "skirts": "#FF8C42",
            "party-dresses": "#F7C948",
            "casual-dresses": "#F7C948",
            "maxi-dresses": "#F7C948",
            "co-ord-sets": "#F7C948",
            "heels": "#9B59B6",
            "boots": "#9B59B6",
            "sneakers": "#9B59B6",
            "loafers": "#9B59B6",
            "formal-shoes": "#9B59B6",
            "sandals": "#9B59B6",
            "flats": "#9B59B6",
            "ethnic-footwear": "#9B59B6",
            "running-shoes": "#9B59B6",
            "suits": "#2ECC71",
            "blazers": "#2ECC71",
            "denim-jackets": "#2ECC71",
            "long-coats": "#2ECC71",
            "nehru-jackets": "#2ECC71",
            "handbags": "#E67E22",
            "clutches": "#E67E22",
            "necklaces": "#E67E22",
            "earrings": "#E67E22",
            "watches": "#E67E22",
            "caps": "#E67E22",
            "sunglasses": "#E67E22",
        }

        node_colors = []
        node_sizes = []
        for nid, ndata in self.G.nodes(data=True):
            cat = ndata.get("category", "")
            c = category_types.get(cat, "#888888")
            if nid == highlight_id:
                c = "#FFFFFF"
            node_colors.append(c)
            # Size proportional to degree
            node_sizes.append(200 + ndata.get("degree", 1) * 150)

        edge_widths = [self.G[u][v].get("weight", 1) * 1.5
                       for u, v in self.G.edges()]

        fig, ax = plt.subplots(figsize=(18, 14), facecolor=DARK)
        ax.set_facecolor(DARK)

        pos = nx.spring_layout(self.G, k=1.5, seed=42, weight="weight")

        nx.draw_networkx_edges(
            self.G, pos, ax=ax,
            width=edge_widths, alpha=0.3, edge_color="#FFFFFF"
        )
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            node_color=node_colors, node_size=node_sizes, alpha=0.9
        )

        # Label only high-degree nodes
        high_degree = {n: ndata["name"][:18]
                       for n, ndata in self.G.nodes(data=True)
                       if ndata.get("degree", 0) >= 3}
        nx.draw_networkx_labels(
            self.G, pos, labels=high_degree, ax=ax,
            font_size=7, font_color="#FFFFFF"
        )

        ax.set_title("Fashion Compatibility Graph\n(nodes = products, edges = expert outfit co-occurrence)",
                     color="#FFFFFF", fontsize=15, fontweight="bold")
        ax.axis("off")

        # Legend
        legend_items = [
            ("Tops/Shirts", "#4F8EF7"), ("T-Shirts/Sweatshirts", "#00D4FF"),
            ("Women Tops", "#FF6B9D"), ("Bottoms", "#E94560"),
            ("Skirts", "#FF8C42"), ("Dresses/Co-ords", "#F7C948"),
            ("Footwear", "#9B59B6"), ("Outerwear", "#2ECC71"),
            ("Accessories/Bags", "#E67E22"),
        ]
        patches = [mpatches.Patch(color=c, label=l) for l, c in legend_items]
        ax.legend(handles=patches, loc="lower left", facecolor=CARD,
                  edgecolor="#333355", labelcolor="#E0E0FF", fontsize=9,
                  ncol=2, framealpha=0.8)

        if output_path:
            plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=DARK)
            print(f"[Graph] ✓ Visualisation saved → {output_path}")
        else:
            plt.show()
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(products_df, outfits_df, save: bool = True) -> FashionCompatibilityGraph:
    """Build the compatibility graph and optionally save to disk."""
    g = FashionCompatibilityGraph()
    g.build(products_df, outfits_df)
    if save:
        g.save()
    return g


def load_graph() -> FashionCompatibilityGraph:
    """Load the pre-built compatibility graph from disk."""
    g = FashionCompatibilityGraph()
    if GRAPH_CACHE.exists():
        g.load()
    else:
        raise FileNotFoundError(
            f"Graph cache not found at {GRAPH_CACHE}. Run build_graph() first."
        )
    return g


if __name__ == "__main__":
    from src.data.loader import load_products, load_outfits

    products = load_products()
    outfits  = load_outfits()

    print("=" * 60)
    print("  PHASE 4: COMPATIBILITY GRAPH TEST")
    print("=" * 60)

    g = build_graph(products, outfits)

    # Test: what pairs with the white formal shirt?
    shirt_id = "myntra_28569210"
    print(f"\nItems compatible with: {g.G.nodes[shirt_id].get('name')}")
    compatibles = g.get_compatible_items(shirt_id, top_k=5)
    for item in compatibles:
        print(f"  [{item['graph_weight']}×] {item['name'][:45]} ({item['category_label']})")

    # Graph score between shirt and grey trousers
    trouser_id = "myntra_23237806"
    score = g.compatibility_score(shirt_id, trouser_id)
    print(f"\nCompatibility score (shirt ↔ trousers): {score:.3f}")

    # Staple items
    print("\nTop men's staple items:")
    for s in g.get_staple_items(gender="men", top_k=5):
        print(f"  [deg={s['degree']}] {s['name'][:45]}")

    # Save graph visualisation
    g.visualise(output_path=str(ROOT / "analysis" / "charts" / "08_compatibility_graph.png"))
