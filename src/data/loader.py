"""
src/data/loader.py
------------------
Core dataset loader for the Dare XAI Fashion AI system.
Handles products.csv, outfits.csv, and provides clean DataFrames
with type normalization, missing value handling, and derived fields.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# ── Path configuration ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # D:\DarexAi
PRODUCTS_PATH = BASE_DIR / "products.csv"
OUTFITS_PATH = BASE_DIR / "outfits.csv"
CURATED_PATH = BASE_DIR / "curated25.xlsx"
IMAGES_DIR = BASE_DIR / "images"


# ── Column constants ──────────────────────────────────────────────────────────
PRODUCT_COLS = [
    "id", "name", "brand", "price_inr", "rating", "rating_count",
    "gender", "wear_type", "category", "category_label", "occasion",
    "tags", "description", "description_source", "image", "site",
    "product_url", "collected_at",
]

OUTFIT_COLS = [
    "outfit_id", "gender", "wear_type", "occasion", "theme",
    "hero", "hero_id", "second", "second_id", "layer", "layer_id",
    "footwear", "footwear_id", "accessory_1", "accessory_1_id",
    "accessory_2", "accessory_2_id", "palette", "items_count",
    "total_price_inr", "image_files", "stylist_rationale",
]


def load_products(validate: bool = True) -> pd.DataFrame:
    """
    Load and preprocess products.csv.

    Returns a clean DataFrame with:
    - Normalised string fields (lowercase, stripped)
    - price_inr as float
    - rating / rating_count as float
    - image_path as absolute Path string
    - tags_list: Python list from semicolon-separated tags
    - has_image: bool indicating if the image file exists on disk
    """
    df = pd.read_csv(PRODUCTS_PATH, dtype=str)

    # ── Numeric coercions ─────────────────────────────────────────────────────
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

    # ── Normalise categorical string fields ───────────────────────────────────
    str_cols = ["gender", "wear_type", "category", "category_label",
                "occasion", "site", "description_source"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    # ── Derived fields ────────────────────────────────────────────────────────
    # Split tags into Python list
    df["tags_list"] = df["tags"].apply(
        lambda t: [x.strip() for x in str(t).split(";")] if pd.notna(t) else []
    )

    # Absolute image path
    df["image_path"] = df["image"].apply(
        lambda p: str(BASE_DIR / p) if pd.notna(p) else None
    )

    # Check if image actually exists on disk
    df["has_image"] = df["image_path"].apply(
        lambda p: os.path.isfile(p) if p else False
    )

    # Source platform (ajio | myntra | nykaa) from id prefix
    df["platform"] = df["id"].apply(
        lambda x: x.split("_")[0] if pd.notna(x) and "_" in str(x) else "unknown"
    )

    # Description quality: site descriptions are richer
    df["rich_description"] = df["description_source"] == "site"

    # Combined text field for embedding
    df["embed_text"] = df.apply(_build_embed_text, axis=1)

    if validate:
        _validate_products(df)

    return df.reset_index(drop=True)


def load_outfits(validate: bool = True) -> pd.DataFrame:
    """
    Load and preprocess outfits.csv.

    Returns a clean DataFrame with:
    - item_ids: dict mapping slot → product_id
    - all_item_ids: flat list of all product IDs in the outfit
    - palette_list: parsed list of palette colours
    - slot_count: number of filled outfit slots
    """
    df = pd.read_csv(OUTFITS_PATH, dtype=str)

    # Normalise strings
    str_cols = ["gender", "wear_type", "occasion", "theme"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    # Numeric conversions
    df["items_count"] = pd.to_numeric(df["items_count"], errors="coerce")
    df["total_price_inr"] = pd.to_numeric(df["total_price_inr"], errors="coerce")

    # ── Derived fields ────────────────────────────────────────────────────────
    # Flat list of all product IDs in each outfit
    id_cols = ["hero_id", "second_id", "layer_id", "footwear_id",
               "accessory_1_id", "accessory_2_id"]
    df["all_item_ids"] = df[id_cols].apply(
        lambda row: [v for v in row.values if pd.notna(v) and str(v).strip() != ""],
        axis=1
    )

    # Slot → id mapping
    df["item_ids"] = df.apply(_extract_item_ids, axis=1)

    # Palette as list
    df["palette_list"] = df["palette"].apply(
        lambda p: [c.strip() for c in str(p).split("/")] if pd.notna(p) else []
    )

    # Image paths list
    df["image_paths"] = df["image_files"].apply(
        lambda imgs: [str(BASE_DIR / p.strip()) for p in str(imgs).split(";")]
        if pd.notna(imgs) else []
    )

    if validate:
        _validate_outfits(df)

    return df.reset_index(drop=True)


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load both datasets."""
    return load_products(), load_outfits()


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_embed_text(row: pd.Series) -> str:
    """
    Build a rich text string for embedding by combining:
    name + brand + category + occasion + wear_type + tags + description.
    """
    parts = []
    for field in ["name", "brand", "category_label", "occasion", "wear_type"]:
        val = row.get(field, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())

    # Add tags (skip duplicates already captured above)
    tags = row.get("tags_list", [])
    if tags:
        parts.append(" ".join(tags))

    # Description (first 200 chars to keep embeddings stable)
    desc = row.get("description", "")
    if pd.notna(desc) and str(desc).strip():
        parts.append(str(desc)[:300])

    return " | ".join(parts)


def _extract_item_ids(row: pd.Series) -> Dict[str, Optional[str]]:
    """Map outfit slot names to product IDs."""
    def _val(col):
        v = row.get(col, "")
        return str(v).strip() if pd.notna(v) and str(v).strip() else None

    return {
        "hero": _val("hero_id"),
        "second": _val("second_id"),
        "layer": _val("layer_id"),
        "footwear": _val("footwear_id"),
        "accessory_1": _val("accessory_1_id"),
        "accessory_2": _val("accessory_2_id"),
    }


def _validate_products(df: pd.DataFrame) -> None:
    """Print a quick validation summary."""
    total = len(df)
    print(f"[Loader] Products loaded: {total}")
    print(f"  Missing price   : {df['price_inr'].isna().sum()}")
    print(f"  Missing rating  : {df['rating'].isna().sum()}")
    print(f"  Images on disk  : {df['has_image'].sum()} / {total}")
    print(f"  Rich desc       : {df['rich_description'].sum()} / {total}")


def _validate_outfits(df: pd.DataFrame) -> None:
    """Print a quick validation summary."""
    total = len(df)
    print(f"[Loader] Outfits loaded: {total}")
    print(f"  Women outfits   : {(df['gender'] == 'women').sum()}")
    print(f"  Men outfits     : {(df['gender'] == 'men').sum()}")
    avg_items = df["items_count"].mean()
    print(f"  Avg items/outfit: {avg_items:.1f}")


# ── Utility: get a product by ID ──────────────────────────────────────────────

def get_product_by_id(products_df: pd.DataFrame, product_id: str) -> Optional[pd.Series]:
    """Return a single product row by ID or None if not found."""
    match = products_df[products_df["id"] == product_id]
    return match.iloc[0] if len(match) > 0 else None


def get_products_by_ids(products_df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    """Return multiple products by ID list, preserving order."""
    return products_df[products_df["id"].isin(ids)].copy()


if __name__ == "__main__":
    products, outfits = load_all()
    print("\nSample product:")
    print(products[["id", "name", "gender", "occasion", "price_inr"]].head(3).to_string())
    print("\nSample outfit:")
    print(outfits[["outfit_id", "theme", "gender", "occasion", "all_item_ids"]].head(3).to_string())
