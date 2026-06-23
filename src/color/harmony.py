"""
src/color/harmony.py
---------------------
Bonus 5: Color Harmony Engine.

Extracts dominant colors from product images using k-means clustering
and applies classical fashion color harmony rules to score item pairings.

Fashion Color Rules:
  1. Monochromatic — same hue, different tones  → safe, always works
  2. Analogous      — adjacent hues on wheel     → natural, calm
  3. Complementary  — opposite hues             → bold, high contrast
  4. Neutral pairing— black/white/grey/beige    → universally compatible
  5. Triadic        — 3 equidistant hues         → vibrant, needs balance
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Named color mapping
# Fashion colours → approximate HSV hue ranges
# ══════════════════════════════════════════════════════════════════════════════

NEUTRAL_COLORS = {"black", "white", "grey", "gray", "beige", "cream", "tan", "nude", "off-white"}

# Approximate hue (0-360) for named fashion colours
COLOR_HUES = {
    "red":      0,
    "maroon":   0,
    "burgundy": 0,
    "orange":   30,
    "coral":    16,
    "yellow":   60,
    "gold":     45,
    "green":    120,
    "olive":    80,
    "teal":     175,
    "cyan":     180,
    "turquoise":175,
    "blue":     210,
    "navy":     220,
    "indigo":   260,
    "purple":   280,
    "violet":   270,
    "pink":     340,
    "rose":     330,
    "magenta":  300,
    "lavender": 270,
    "brown":    25,
    "rust":     15,
    "khaki":    55,
}


def hue_distance(h1: float, h2: float) -> float:
    """Circular distance between two hues [0-360]."""
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)


def color_harmony_score(colors_a: List[str], colors_b: List[str]) -> Dict:
    """
    Score the colour harmony between two items based on their named colours.

    Returns dict with:
      score: float [0, 1]
      rule: str (which harmony rule matched)
      explanation: str
    """
    # Separate neutrals from chromatic colours
    chromatic_a = [c for c in colors_a if c not in NEUTRAL_COLORS]
    chromatic_b = [c for c in colors_b if c not in NEUTRAL_COLORS]
    neutral_a   = [c for c in colors_a if c in NEUTRAL_COLORS]
    neutral_b   = [c for c in colors_b if c in NEUTRAL_COLORS]

    # ── Rule 0: Neutral + anything = great ────────────────────────────────────
    if not chromatic_a or not chromatic_b:
        return {
            "score": 0.90,
            "rule": "neutral",
            "explanation": "Neutral tones pair universally well with any colour"
        }

    # Get hues for chromatic colours
    hues_a = [COLOR_HUES.get(c, None) for c in chromatic_a]
    hues_b = [COLOR_HUES.get(c, None) for c in chromatic_b]
    hues_a = [h for h in hues_a if h is not None]
    hues_b = [h for h in hues_b if h is not None]

    if not hues_a or not hues_b:
        return {"score": 0.6, "rule": "unknown", "explanation": "Colour data insufficient"}

    # Take primary hue (first chromatic)
    h_a = hues_a[0]
    h_b = hues_b[0]
    dist = hue_distance(h_a, h_b)

    # ── Rule 1: Monochromatic (0-30°) ─────────────────────────────────────────
    if dist <= 30:
        return {
            "score": 0.82,
            "rule": "monochromatic",
            "explanation": f"Tonal harmony — both items share a similar colour family, "
                           f"creating a cohesive, polished look"
        }

    # ── Rule 2: Analogous (30-60°) ────────────────────────────────────────────
    if dist <= 60:
        return {
            "score": 0.78,
            "rule": "analogous",
            "explanation": f"Analogous colour pairing — adjacent hues create "
                           f"a natural, harmonious combination"
        }

    # ── Rule 3: Complementary (150-210°) ─────────────────────────────────────
    if 150 <= dist <= 210:
        return {
            "score": 0.88,
            "rule": "complementary",
            "explanation": f"Complementary contrast — opposite hues create "
                           f"a bold, high-impact visual statement"
        }

    # ── Rule 4: Triadic (110-130°) ────────────────────────────────────────────
    if 110 <= dist <= 130:
        return {
            "score": 0.72,
            "rule": "triadic",
            "explanation": f"Triadic colour play — vibrant combination that "
                           f"works best when one colour dominates"
        }

    # ── Default: works but needs care ─────────────────────────────────────────
    return {
        "score": 0.55,
        "rule": "split",
        "explanation": f"An interesting colour combination — "
                       f"use accessories to bridge the tones"
    }


def extract_dominant_colors_from_image(image_path: str, n_colors: int = 3) -> List[str]:
    """
    Extract dominant colors from a product image using k-means clustering.
    Returns list of approximate named colors.
    """
    try:
        from PIL import Image
        from sklearn.cluster import KMeans

        img = Image.open(image_path).convert("RGB")
        img = img.resize((100, 100))  # Downscale for speed
        pixels = np.array(img).reshape(-1, 3).astype(np.float32)

        # K-means clustering
        km = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
        km.fit(pixels)

        # Map cluster centers to named colours
        named = []
        for center in km.cluster_centers_:
            r, g, b = int(center[0]), int(center[1]), int(center[2])
            name = _rgb_to_name(r, g, b)
            named.append(name)

        return named

    except Exception as e:
        return []


def _rgb_to_name(r: int, g: int, b: int) -> str:
    """Map RGB values to the nearest fashion colour name."""
    # Convert to HSV
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r_, g_, b_)
    min_c = min(r_, g_, b_)
    delta = max_c - min_c

    # Achromatic (neutral)
    if delta < 0.12:
        if max_c < 0.15:
            return "black"
        if max_c > 0.85:
            return "white"
        if max_c > 0.70:
            return "cream"
        return "grey"

    # Beige / tan / brown
    if r_ > g_ > b_ and delta < 0.4:
        if max_c < 0.55:
            return "brown"
        return "beige" if max_c > 0.7 else "tan"

    # Compute hue
    if max_c == r_:
        hue = 60 * (((g_ - b_) / delta) % 6)
    elif max_c == g_:
        hue = 60 * (((b_ - r_) / delta) + 2)
    else:
        hue = 60 * (((r_ - g_) / delta) + 4)
    if hue < 0:
        hue += 360

    sat = delta / max_c if max_c > 0 else 0
    val = max_c

    # Dark colours → navy, maroon, olive
    if val < 0.25:
        if hue < 30 or hue > 330:
            return "maroon"
        if 200 <= hue <= 260:
            return "navy"
        if 60 <= hue <= 100:
            return "olive"
        return "dark"

    # Map hue to colour name
    if hue < 20 or hue >= 345:
        return "red"
    if hue < 40:
        return "orange" if sat > 0.7 else "brown"
    if hue < 65:
        return "gold" if val > 0.7 else "olive"
    if hue < 80:
        return "olive"
    if hue < 150:
        return "green"
    if hue < 190:
        return "teal"
    if hue < 250:
        return "blue" if hue > 210 else "navy"
    if hue < 290:
        return "purple"
    if hue < 330:
        return "pink"
    return "red"


def score_outfit_color_harmony(items: List[Dict]) -> Dict:
    """
    Score colour harmony for a complete outfit.
    Uses named colours extracted from item names/tags as proxy for images.

    Returns: {score: float, rule: str, explanation: str}
    """
    color_keywords = ["black", "white", "navy", "blue", "red", "green",
                      "brown", "grey", "cream", "gold", "pink", "maroon",
                      "olive", "beige", "purple", "burgundy", "coral",
                      "teal", "orange", "rust", "tan", "cyan", "lavender"]

    def extract_text_colors(item: Dict) -> List[str]:
        text = (item.get("name", "") + " " + item.get("tags", "")).lower()
        return [c for c in color_keywords if c in text]

    if len(items) < 2:
        return {"score": 0.7, "rule": "single", "explanation": "Single item — no pairing to evaluate"}

    # Score all pairs and average
    pair_scores = []
    from itertools import combinations
    for ia, ib in combinations(items, 2):
        ca = extract_text_colors(ia)
        cb = extract_text_colors(ib)
        if ca and cb:
            s = color_harmony_score(ca, cb)
            pair_scores.append(s)

    if not pair_scores:
        return {"score": 0.65, "rule": "unknown", "explanation": "Could not extract colour data"}

    avg_score = np.mean([s["score"] for s in pair_scores])
    # Use the most common rule
    from collections import Counter
    rule_counts = Counter(s["rule"] for s in pair_scores)
    dominant_rule = rule_counts.most_common(1)[0][0]
    explanation = next(s["explanation"] for s in pair_scores if s["rule"] == dominant_rule)

    return {
        "score": float(avg_score),
        "rule": dominant_rule,
        "explanation": explanation,
        "percentage": int(avg_score * 100),
    }


if __name__ == "__main__":
    # Test colour harmony rules
    pairs = [
        (["navy"], ["white"], "Navy + White (complementary)"),
        (["black"], ["red"], "Black + Red (high contrast)"),
        (["olive"], ["brown"], "Olive + Brown (analogous)"),
        (["grey"], ["pink"], "Grey + Pink (neutral + chromatic)"),
        (["navy"], ["grey"], "Navy + Grey (analogous dark)"),
    ]
    print("Color Harmony Test:")
    for c_a, c_b, desc in pairs:
        result = color_harmony_score(c_a, c_b)
        print(f"  {desc}: score={result['score']:.2f} rule={result['rule']}")
