"""
analysis/phase1_analysis.py
----------------------------
Phase 1: Full Dataset Analysis for the Dare XAI Fashion AI System.

Generates:
  1. Dataset summary stats
  2. Product distribution (gender, category, occasion, platform, price, rating)
  3. Category distribution chart
  4. Occasion distribution chart
  5. Gender distribution chart
  6. Missing values analysis
  7. Metadata quality assessment
  8. Palette analysis (from outfits)
  9. Outfit relationship analysis (co-occurrence, staple items)
 10. Combined analysis dashboard
 11. Saves JSON report for later use

Run from D:\DarexAi root:
    venv\Scripts\python analysis\phase1_analysis.py
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_products, load_outfits

# ── Output directories ────────────────────────────────────────────────────────
CHARTS_DIR = ROOT / "analysis" / "charts"
REPORTS_DIR = ROOT / "analysis" / "reports"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
PALETTE_DARK   = "#0F0F1A"
PALETTE_CARD   = "#1A1A2E"
PALETTE_ACCENT = "#E94560"
PALETTE_BLUE   = "#4F8EF7"
PALETTE_GOLD   = "#F7C948"
PALETTE_GREEN  = "#2ECC71"
PALETTE_PURPLE = "#9B59B6"
PALETTE_CORAL  = "#E67E22"

BRAND_COLORS = [PALETTE_ACCENT, PALETTE_BLUE, PALETTE_GOLD, PALETTE_GREEN,
                PALETTE_PURPLE, PALETTE_CORAL, "#00D4FF", "#FF6B9D"]

plt.rcParams.update({
    "figure.facecolor": PALETTE_DARK,
    "axes.facecolor": PALETTE_CARD,
    "axes.edgecolor": "#333355",
    "axes.labelcolor": "#E0E0FF",
    "xtick.color": "#B0B0CC",
    "ytick.color": "#B0B0CC",
    "text.color": "#E0E0FF",
    "grid.color": "#2A2A4A",
    "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1 ─ Load data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  PHASE 1: DATASET ANALYSIS — Dare XAI Fashion AI")
print("=" * 60)

products = load_products()
outfits  = load_outfits()

print(f"\n✓ Loaded {len(products)} products")
print(f"✓ Loaded {len(outfits)} outfits\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2 ─ Dataset Summary
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 40)
print("DATASET SUMMARY")
print("─" * 40)

summary = {
    "total_products": len(products),
    "total_outfits": len(outfits),
    "platforms": products["platform"].value_counts().to_dict(),
    "gender_split": products["gender"].value_counts().to_dict(),
    "wear_type_split": products["wear_type"].value_counts().to_dict(),
    "unique_categories": products["category"].nunique(),
    "unique_occasions": products["occasion"].nunique(),
    "unique_brands": products["brand"].nunique(),
    "price_stats": {
        "min": float(products["price_inr"].min()),
        "max": float(products["price_inr"].max()),
        "mean": float(products["price_inr"].mean()),
        "median": float(products["price_inr"].median()),
    },
    "rating_stats": {
        "min": float(products["rating"].min()),
        "max": float(products["rating"].max()),
        "mean": float(products["rating"].mean()),
    },
    "missing_values": products.isnull().sum().to_dict(),
    "images_on_disk": int(products["has_image"].sum()),
    "rich_descriptions": int(products["rich_description"].sum()),
}

print(json.dumps({k: v for k, v in summary.items() if k not in ["missing_values"]}, indent=2))

# Save full summary
with open(REPORTS_DIR / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Summary saved → analysis/reports/dataset_summary.json")


# ══════════════════════════════════════════════════════════════════════════════
# 3 ─ Chart 1: Overview Dashboard (2x3 grid)
# ══════════════════════════════════════════════════════════════════════════════
print("\n─ Generating Chart 1: Overview Dashboard...")

fig = plt.figure(figsize=(20, 14), facecolor=PALETTE_DARK)
fig.suptitle("Dare XAI Fashion Dataset — Analysis Overview",
             fontsize=20, fontweight="bold", color="#FFFFFF", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 3a: Gender distribution (donut) ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
gender_counts = products["gender"].value_counts()
colors_g = [PALETTE_BLUE, PALETTE_ACCENT]
wedges, texts, autotexts = ax1.pie(
    gender_counts.values,
    labels=gender_counts.index.str.title(),
    autopct="%1.0f%%",
    startangle=90,
    colors=colors_g,
    pctdistance=0.75,
    wedgeprops={"linewidth": 2, "edgecolor": PALETTE_DARK, "width": 0.5},
)
for t in texts:
    t.set_color("#E0E0FF")
    t.set_fontsize(12)
for at in autotexts:
    at.set_color("#FFFFFF")
    at.set_fontweight("bold")
ax1.set_title("Gender Distribution", color="#FFFFFF", fontweight="bold", pad=10)
# Centre label
ax1.text(0, 0, f"{len(products)}\nProducts",
         ha="center", va="center", color="#FFFFFF",
         fontsize=14, fontweight="bold")

# ── 3b: Platform distribution ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
platform_counts = products["platform"].value_counts()
bars = ax2.bar(platform_counts.index.str.title(), platform_counts.values,
               color=[PALETTE_ACCENT, PALETTE_BLUE, PALETTE_GOLD], width=0.5,
               edgecolor=PALETTE_DARK, linewidth=1.5)
for bar, val in zip(bars, platform_counts.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             str(val), ha="center", va="bottom", color="#FFFFFF", fontweight="bold")
ax2.set_title("Products by Platform", color="#FFFFFF", fontweight="bold")
ax2.set_ylabel("Count")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, platform_counts.max() + 5)

# ── 3c: Wear type distribution ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
wear_counts = products["wear_type"].value_counts()
colors_w = [PALETTE_GREEN, PALETTE_PURPLE, PALETTE_GOLD, PALETTE_CORAL, PALETTE_BLUE][:len(wear_counts)]
wedges3, texts3, autos3 = ax3.pie(
    wear_counts.values,
    labels=wear_counts.index.str.title(),
    autopct="%1.0f%%",
    colors=colors_w,
    startangle=45,
    wedgeprops={"linewidth": 2, "edgecolor": PALETTE_DARK},
    pctdistance=0.80,
)
for t in texts3:
    t.set_color("#E0E0FF")
for at in autos3:
    at.set_color("#FFFFFF")
    at.set_fontweight("bold")
ax3.set_title("Wear Type Distribution", color="#FFFFFF", fontweight="bold")

# ── 3d: Occasion distribution ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
occ_counts = products["occasion"].value_counts()
colors_o = BRAND_COLORS[:len(occ_counts)]
bars4 = ax4.barh(occ_counts.index.str.title(), occ_counts.values,
                 color=colors_o, edgecolor=PALETTE_DARK, linewidth=1)
for bar, val in zip(bars4, occ_counts.values):
    ax4.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
             str(val), va="center", color="#FFFFFF", fontweight="bold", fontsize=10)
ax4.set_title("Occasion Distribution", color="#FFFFFF", fontweight="bold")
ax4.set_xlabel("Count")
ax4.grid(axis="x", alpha=0.3)
ax4.set_xlim(0, occ_counts.max() + 3)

# ── 3e: Price distribution ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
price_data = products["price_inr"].dropna()
ax5.hist(price_data, bins=15, color=PALETTE_BLUE, edgecolor=PALETTE_DARK,
         linewidth=1, alpha=0.85)
ax5.axvline(price_data.median(), color=PALETTE_GOLD, linestyle="--",
            linewidth=2, label=f"Median ₹{price_data.median():.0f}")
ax5.axvline(price_data.mean(), color=PALETTE_ACCENT, linestyle="--",
            linewidth=2, label=f"Mean ₹{price_data.mean():.0f}")
ax5.legend(facecolor=PALETTE_CARD, edgecolor="#333355", labelcolor="#E0E0FF")
ax5.set_title("Price Distribution (₹)", color="#FFFFFF", fontweight="bold")
ax5.set_xlabel("Price (INR)")
ax5.set_ylabel("Count")
ax5.grid(alpha=0.3)

# ── 3f: Rating distribution ───────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
rating_data = products["rating"].dropna()
ax6.hist(rating_data, bins=10, color=PALETTE_GREEN, edgecolor=PALETTE_DARK,
         linewidth=1, alpha=0.85)
ax6.axvline(rating_data.mean(), color=PALETTE_GOLD, linestyle="--",
            linewidth=2, label=f"Mean {rating_data.mean():.2f}★")
ax6.legend(facecolor=PALETTE_CARD, edgecolor="#333355", labelcolor="#E0E0FF")
ax6.set_title("Rating Distribution", color="#FFFFFF", fontweight="bold")
ax6.set_xlabel("Rating (out of 5)")
ax6.set_ylabel("Count")
ax6.grid(alpha=0.3)
# Note missing ratings
missing_r = products["rating"].isna().sum()
ax6.text(0.98, 0.95, f"{missing_r} products\nno rating",
         transform=ax6.transAxes, ha="right", va="top",
         color=PALETTE_CORAL, fontsize=9)

plt.savefig(CHARTS_DIR / "01_overview_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE_DARK)
plt.close()
print("  ✓ Chart 1 saved → analysis/charts/01_overview_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4 ─ Chart 2: Category Distribution (full breakdown)
# ══════════════════════════════════════════════════════════════════════════════
print("─ Generating Chart 2: Category Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor=PALETTE_DARK)
fig.suptitle("Category Distribution by Gender",
             fontsize=18, fontweight="bold", color="#FFFFFF", y=1.01)

for ax, gender in zip(axes, ["men", "women"]):
    g_data = products[products["gender"] == gender]
    cat_counts = g_data["category_label"].value_counts()
    color = PALETTE_BLUE if gender == "men" else PALETTE_ACCENT
    cmap = LinearSegmentedColormap.from_list("custom", [PALETTE_CARD, color])
    bar_colors = [cmap(i / max(len(cat_counts) - 1, 1)) for i in range(len(cat_counts))]

    bars = ax.barh(cat_counts.index, cat_counts.values,
                   color=bar_colors, edgecolor=PALETTE_DARK, linewidth=1)
    for bar, val in zip(bars, cat_counts.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color="#FFFFFF", fontweight="bold", fontsize=10)

    ax.set_title(f"{gender.title()} Categories", color="#FFFFFF",
                 fontweight="bold", fontsize=14)
    ax.set_xlabel("Number of Products")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, cat_counts.max() + 2)
    # Total label
    ax.text(0.98, 0.02, f"Total: {len(g_data)} items",
            transform=ax.transAxes, ha="right", va="bottom",
            color=color, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(CHARTS_DIR / "02_category_distribution.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE_DARK)
plt.close()
print("  ✓ Chart 2 saved → analysis/charts/02_category_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5 ─ Chart 3: Missing Values Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("─ Generating Chart 3: Missing Values Analysis...")

core_cols = ["id", "name", "brand", "price_inr", "rating", "rating_count",
             "gender", "wear_type", "category", "occasion", "tags", "description", "image"]
missing_df = products[core_cols].isnull()
missing_pct = (missing_df.sum() / len(products) * 100).reset_index()
missing_pct.columns = ["Field", "Missing %"]
missing_pct = missing_pct.sort_values("Missing %", ascending=False)

fig, (ax_bar, ax_table) = plt.subplots(1, 2, figsize=(18, 7), facecolor=PALETTE_DARK)
fig.suptitle("Missing Values & Metadata Quality Analysis",
             fontsize=17, fontweight="bold", color="#FFFFFF")

# Bar chart
color_scale = [PALETTE_ACCENT if v > 20 else PALETTE_GOLD if v > 5 else PALETTE_GREEN
               for v in missing_pct["Missing %"]]
bars = ax_bar.barh(missing_pct["Field"], missing_pct["Missing %"],
                   color=color_scale, edgecolor=PALETTE_DARK)
for bar, val in zip(bars, missing_pct["Missing %"]):
    ax_bar.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#FFFFFF", fontsize=10)
ax_bar.set_title("Missing Values by Field", color="#FFFFFF", fontweight="bold")
ax_bar.set_xlabel("Missing (%)")
ax_bar.axvline(5, color=PALETTE_GOLD, linestyle="--", alpha=0.5, label=">5% warning")
ax_bar.axvline(20, color=PALETTE_ACCENT, linestyle="--", alpha=0.5, label=">20% critical")
ax_bar.legend(facecolor=PALETTE_CARD, edgecolor="#333355", labelcolor="#E0E0FF", fontsize=9)
ax_bar.grid(axis="x", alpha=0.3)

# Description quality table
desc_quality = {
    "site (rich)": int((products["description_source"] == "site").sum()),
    "metadata (short)": int((products["description_source"] == "metadata").sum()),
    "Images on disk": int(products["has_image"].sum()),
    "Images missing": int((~products["has_image"]).sum()),
}
ax_table.axis("off")
table_data = [[k, str(v), f"{v/len(products)*100:.0f}%"] for k, v in desc_quality.items()]
table = ax_table.table(
    cellText=table_data,
    colLabels=["Metadata Field", "Count", "Coverage"],
    cellLoc="center", loc="center",
    bbox=[0.05, 0.1, 0.9, 0.8],
)
table.auto_set_font_size(False)
table.set_fontsize(12)
for (row, col), cell in table.get_celld().items():
    cell.set_facecolor(PALETTE_CARD if row > 0 else PALETTE_ACCENT)
    cell.set_edgecolor("#333355")
    cell.set_text_props(color="#FFFFFF")
ax_table.set_title("Data Quality Metrics", color="#FFFFFF", fontweight="bold")

plt.tight_layout()
plt.savefig(CHARTS_DIR / "03_missing_values.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE_DARK)
plt.close()
print("  ✓ Chart 3 saved → analysis/charts/03_missing_values.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6 ─ Chart 4: Outfit Palette Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("─ Generating Chart 4: Palette Analysis...")

# Flatten all palette colours
all_colors = []
for palette_list in outfits["palette_list"]:
    all_colors.extend([c.strip().lower() for c in palette_list if c.strip()])

color_counts = Counter(all_colors)
top_colors = dict(sorted(color_counts.items(), key=lambda x: -x[1])[:15])

# Map colour names to hex (best-effort)
COLOR_HEX = {
    "black": "#1A1A1A", "white": "#F5F5F5", "navy": "#1B3A6B",
    "blue": "#3B82F6", "red": "#EF4444", "grey": "#6B7280",
    "green": "#22C55E", "brown": "#92400E", "olive": "#6B7C3A",
    "maroon": "#7F1D1D", "cream": "#FEF3C7", "gold": "#F59E0B",
    "purple": "#7C3AED", "pink": "#EC4899", "tan": "#D97706",
    "burgundy": "#7F1D4F", "beige": "#D4B896",
}
bar_colors_palette = [COLOR_HEX.get(c, PALETTE_BLUE) for c in top_colors.keys()]

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7), facecolor=PALETTE_DARK)
fig.suptitle("Colour Palette Analysis Across 25 Expert-Curated Outfits",
             fontsize=16, fontweight="bold", color="#FFFFFF")

# Frequency chart
bars = ax_left.bar(list(top_colors.keys()), list(top_colors.values()),
                   color=bar_colors_palette, edgecolor=PALETTE_DARK, linewidth=1)
for bar, val in zip(bars, top_colors.values()):
    ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(val), ha="center", va="bottom", color="#FFFFFF",
                 fontsize=9, fontweight="bold")
ax_left.set_title("Most Used Colours in Outfits", color="#FFFFFF", fontweight="bold")
ax_left.set_ylabel("Frequency")
ax_left.tick_params(axis="x", rotation=45)
ax_left.grid(axis="y", alpha=0.3)

# Colour swatches
ax_right.axis("off")
ax_right.set_title("Colour Palette Reference", color="#FFFFFF", fontweight="bold")
swatch_cols = 4
items = list(top_colors.items())
for i, (color_name, count) in enumerate(items):
    row, col = divmod(i, swatch_cols)
    x = col * 0.25 + 0.02
    y = 0.85 - row * 0.18
    hex_c = COLOR_HEX.get(color_name, "#888888")
    rect = mpatches.FancyBboxPatch(
        (x, y), 0.18, 0.12,
        boxstyle="round,pad=0.01",
        facecolor=hex_c,
        edgecolor="#FFFFFF" if color_name == "white" else hex_c,
        linewidth=1,
        transform=ax_right.transAxes,
    )
    ax_right.add_patch(rect)
    ax_right.text(x + 0.09, y + 0.06, f"{color_name.title()}\n({count}×)",
                  ha="center", va="center", transform=ax_right.transAxes,
                  color="black" if color_name in ("white", "cream", "gold") else "white",
                  fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(CHARTS_DIR / "04_palette_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE_DARK)
plt.close()
print("  ✓ Chart 4 saved → analysis/charts/04_palette_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7 ─ Chart 5: Outfit Relationship Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("─ Generating Chart 5: Outfit Relationship Analysis...")

# Count product appearances across outfits
product_appearances = Counter()
for ids in outfits["all_item_ids"]:
    product_appearances.update(ids)

# Top "staple" items
top_staples = product_appearances.most_common(12)
staple_ids = [pid for pid, _ in top_staples]
staple_counts = [cnt for _, cnt in top_staples]

# Get product names for labels
id_to_name = dict(zip(products["id"], products["name"]))
id_to_cat = dict(zip(products["id"], products["category_label"]))
staple_labels = [
    f"{id_to_name.get(pid, pid)[:25]}\n({id_to_cat.get(pid, '?')})"
    for pid in staple_ids
]

fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor=PALETTE_DARK)
fig.suptitle("Outfit Relationship & Staple Item Analysis",
             fontsize=16, fontweight="bold", color="#FFFFFF")

# Staple items chart
color_map = BRAND_COLORS[:len(top_staples)]
bars = axes[0].barh(staple_labels[::-1], staple_counts[::-1],
                    color=color_map[::-1], edgecolor=PALETTE_DARK, linewidth=1)
for bar, val in zip(bars, staple_counts[::-1]):
    axes[0].text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val}×", va="center", color="#FFFFFF",
                 fontweight="bold", fontsize=10)
axes[0].set_title("Most Used Products Across Outfits\n(Staple Items)",
                  color="#FFFFFF", fontweight="bold")
axes[0].set_xlabel("# Outfits Featuring This Product")
axes[0].grid(axis="x", alpha=0.3)

# Outfit slot analysis
slot_cols = ["hero_id", "second_id", "layer_id", "footwear_id",
             "accessory_1_id", "accessory_2_id"]
slot_names = ["Hero", "Second", "Layer", "Footwear", "Accessory 1", "Accessory 2"]
slot_fill = [(outfits[col].notna() & (outfits[col].str.strip() != "")).sum()
             for col in slot_cols]

colors_slot = [PALETTE_ACCENT, PALETTE_BLUE, PALETTE_GREEN,
               PALETTE_GOLD, PALETTE_PURPLE, PALETTE_CORAL]
wedges, texts, autotexts = axes[1].pie(
    slot_fill, labels=slot_names, autopct="%1.0f%%",
    colors=colors_slot, startangle=90,
    wedgeprops={"linewidth": 2, "edgecolor": PALETTE_DARK},
    pctdistance=0.80,
)
for t in texts:
    t.set_color("#E0E0FF")
for at in autotexts:
    at.set_color("#FFFFFF")
    at.set_fontweight("bold")
axes[1].set_title("Outfit Slot Fill Rate\n(out of 25 outfits)",
                  color="#FFFFFF", fontweight="bold")

# Add actual numbers
legend_labels = [f"{name}: {cnt}/25" for name, cnt in zip(slot_names, slot_fill)]
legend_patches = [mpatches.Patch(color=c, label=l)
                  for c, l in zip(colors_slot, legend_labels)]
axes[1].legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.15), ncol=2,
               facecolor=PALETTE_CARD, edgecolor="#333355",
               labelcolor="#E0E0FF", fontsize=9)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "05_outfit_relationships.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE_DARK)
plt.close()
print("  ✓ Chart 5 saved → analysis/charts/05_outfit_relationships.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8 ─ Chart 6: Occasion × Gender × Wear-Type Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("─ Generating Chart 6: Occasion × Category Heatmap...")

pivot = products.pivot_table(
    index="occasion", columns="gender",
    values="id", aggfunc="count", fill_value=0
)

fig, (ax_heat, ax_outfit) = plt.subplots(1, 2, figsize=(18, 7), facecolor=PALETTE_DARK)
fig.suptitle("Occasion × Gender Heatmap  |  Outfit Distribution",
             fontsize=16, fontweight="bold", color="#FFFFFF")

# Heatmap
cmap = LinearSegmentedColormap.from_list("heat", [PALETTE_DARK, PALETTE_BLUE, PALETTE_ACCENT])
sns.heatmap(pivot, ax=ax_heat, cmap=cmap, annot=True, fmt="d",
            linewidths=1, linecolor="#0F0F1A",
            annot_kws={"color": "#FFFFFF", "fontweight": "bold", "fontsize": 13},
            cbar_kws={"shrink": 0.7})
ax_heat.set_title("Product Count: Occasion × Gender", color="#FFFFFF", fontweight="bold")
ax_heat.tick_params(colors="#E0E0FF")
ax_heat.set_xlabel("Gender", color="#E0E0FF")
ax_heat.set_ylabel("Occasion", color="#E0E0FF")
ax_heat.set_xticklabels(ax_heat.get_xticklabels(), color="#E0E0FF")
ax_heat.set_yticklabels(ax_heat.get_yticklabels(), color="#E0E0FF", rotation=0)

# Outfit distribution
outfit_occ = outfits["occasion"].value_counts()
outfit_gender = outfits["gender"].value_counts()

x_out = np.arange(len(outfit_occ))
bars_out = ax_outfit.bar(outfit_occ.index.str.title(), outfit_occ.values,
                         color=BRAND_COLORS[:len(outfit_occ)], edgecolor=PALETTE_DARK, linewidth=1)
for bar, val in zip(bars_out, outfit_occ.values):
    ax_outfit.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                   str(val), ha="center", va="bottom", color="#FFFFFF",
                   fontweight="bold", fontsize=11)
ax_outfit.set_title("Outfit Count by Occasion (25 total)", color="#FFFFFF", fontweight="bold")
ax_outfit.set_ylabel("Count")
ax_outfit.tick_params(axis="x", rotation=30)
ax_outfit.grid(axis="y", alpha=0.3)

# Add gender split annotation
m = int(outfit_gender.get("men", 0))
w = int(outfit_gender.get("women", 0))
ax_outfit.text(0.98, 0.97, f"Women: {w}  Men: {m}",
               transform=ax_outfit.transAxes, ha="right", va="top",
               color="#E0E0FF", fontsize=11,
               bbox=dict(boxstyle="round", facecolor=PALETTE_ACCENT, alpha=0.3))

plt.tight_layout()
plt.savefig(CHARTS_DIR / "06_occasion_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE_DARK)
plt.close()
print("  ✓ Chart 6 saved → analysis/charts/06_occasion_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9 ─ Chart 7: Price Range by Category (Box plot style)
# ══════════════════════════════════════════════════════════════════════════════
print("─ Generating Chart 7: Price Range by Category...")

fig, ax = plt.subplots(figsize=(18, 9), facecolor=PALETTE_DARK)
ax.set_facecolor(PALETTE_CARD)

cat_price = products.dropna(subset=["price_inr"]).groupby("category_label")["price_inr"]
cat_means = cat_price.mean().sort_values(ascending=False)
cat_stds = cat_price.std().fillna(0)
sorted_cats = cat_means.index.tolist()

x_pos = np.arange(len(sorted_cats))
means = cat_means.values
stds = cat_stds.reindex(sorted_cats).values

# Color by price level
bar_colors_price = []
for m in means:
    if m > 5000:
        bar_colors_price.append(PALETTE_GOLD)
    elif m > 2000:
        bar_colors_price.append(PALETTE_BLUE)
    else:
        bar_colors_price.append(PALETTE_GREEN)

bars = ax.bar(x_pos, means, color=bar_colors_price,
              edgecolor=PALETTE_DARK, linewidth=1, alpha=0.9)
ax.errorbar(x_pos, means, yerr=stds, fmt="none",
            color=PALETTE_ACCENT, capsize=5, linewidth=1.5)

for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"₹{val:.0f}", ha="center", va="bottom",
            color="#FFFFFF", fontsize=8, rotation=45)

ax.set_xticks(x_pos)
ax.set_xticklabels(sorted_cats, rotation=50, ha="right", color="#E0E0FF", fontsize=9)
ax.set_title("Average Price (₹) by Product Category",
             color="#FFFFFF", fontweight="bold", fontsize=14)
ax.set_ylabel("Price (INR)")
ax.grid(axis="y", alpha=0.3)

legend_patches = [
    mpatches.Patch(color=PALETTE_GOLD, label="> ₹5000 (Premium)"),
    mpatches.Patch(color=PALETTE_BLUE, label="₹2000–5000 (Mid-range)"),
    mpatches.Patch(color=PALETTE_GREEN, label="< ₹2000 (Affordable)"),
]
ax.legend(handles=legend_patches, facecolor=PALETTE_CARD,
          edgecolor="#333355", labelcolor="#E0E0FF", fontsize=10)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "07_price_by_category.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE_DARK)
plt.close()
print("  ✓ Chart 7 saved → analysis/charts/07_price_by_category.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10 ─ Save JSON analysis report
# ══════════════════════════════════════════════════════════════════════════════
print("\n─ Saving analysis report JSON...")

# Top staples with full info
staple_report = []
for pid, count in top_staples:
    prod = products[products["id"] == pid]
    if len(prod):
        row = prod.iloc[0]
        staple_report.append({
            "id": pid,
            "name": row["name"],
            "category": row["category_label"],
            "occasions": row["occasion"],
            "outfit_appearances": count,
        })

report = {
    "phase": "Phase 1 — Dataset Analysis",
    "summary": summary,
    "top_staple_items": staple_report,
    "category_distribution": products["category_label"].value_counts().to_dict(),
    "occasion_distribution": products["occasion"].value_counts().to_dict(),
    "gender_distribution": products["gender"].value_counts().to_dict(),
    "wear_type_distribution": products["wear_type"].value_counts().to_dict(),
    "palette_frequency": dict(color_counts.most_common(20)),
    "outfit_slot_fill": dict(zip(slot_names, slot_fill)),
    "outfit_occasion_distribution": outfits["occasion"].value_counts().to_dict(),
    "charts_generated": [
        "01_overview_dashboard.png",
        "02_category_distribution.png",
        "03_missing_values.png",
        "04_palette_analysis.png",
        "05_outfit_relationships.png",
        "06_occasion_heatmap.png",
        "07_price_by_category.png",
    ],
}

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(REPORTS_DIR / "phase1_full_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, cls=_NpEncoder)
print("  ✓ Report saved → analysis/reports/phase1_full_report.json")


# ══════════════════════════════════════════════════════════════════════════════
# 11 ─ Print final textual summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 1 COMPLETE — KEY FINDINGS")
print("=" * 60)
print(f"\n📦 Dataset Scale")
print(f"   • {len(products)} unique products  |  {len(outfits)} curated outfits")
print(f"   • {products['brand'].nunique()} distinct brands")
print(f"   • {products['category_label'].nunique()} product categories")
print(f"   • Images available: {products['has_image'].sum()}/68")

print(f"\n👥 Gender Split")
print(f"   • {(products['gender']=='women').sum()} women's items  |  {(products['gender']=='men').sum()} men's items")
print(f"   • {(outfits['gender']=='women').sum()} women's outfits  |  {(outfits['gender']=='men').sum()} men's outfits")

print(f"\n💰 Price Range")
print(f"   • ₹{products['price_inr'].min():.0f} – ₹{products['price_inr'].max():.0f}")
print(f"   • Median ₹{products['price_inr'].median():.0f}  |  Mean ₹{products['price_inr'].mean():.0f}")

print(f"\n⭐ Ratings")
print(f"   • {products['rating'].isna().sum()} products lack ratings ({products['rating'].isna().sum()/len(products)*100:.0f}%)")
print(f"   • Average rating: {products['rating'].mean():.2f}/5")

print(f"\n🎨 Palette Insights")
top3 = [c for c, _ in color_counts.most_common(3)]
print(f"   • Top 3 colours: {', '.join(top3)}")
print(f"   • {len(set(all_colors))} unique palette colours used")

print(f"\n🔁 Staple Items (appear in 2+ outfits)")
for item in staple_report[:5]:
    print(f"   • {item['name'][:40]} — {item['outfit_appearances']}× outfits")

print(f"\n⚠️  Metadata Challenges")
print(f"   • rating_count: {products['rating_count'].isna().sum()} missing ({products['rating_count'].isna().sum()/len(products)*100:.0f}%)")
print(f"   • rating: {products['rating'].isna().sum()} missing ({products['rating'].isna().sum()/len(products)*100:.0f}%)")
print(f"   • {(products['description_source']=='metadata').sum()} products have short auto-generated descriptions")
print(f"     → May reduce embedding quality; will compensate with tags + category")

print(f"\n📊 Charts saved in: analysis/charts/")
print(f"📋 Report saved in: analysis/reports/")
print("\nReady for Phase 2: Embedding Generation 🚀\n")
