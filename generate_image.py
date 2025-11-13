"""Generates a grid visualization of problem statuses based on a JSON file.

This script reads problem categories from "status.json" and generates an image
file "status_grid.png" showing a colored grid where each cell represents a
problem and its color indicates its status (e.g., world record, former record).
"""

import json
import math

from matplotlib import patches
import matplotlib.pyplot as plt


# --- Configuration ---
TOTAL_PROBLEMS = 67

# Load data from JSON file
with open("status.json", "r") as f:
  CATEGORIES = json.load(f)

COLORS = {
    "world_record": "#22c55e",
    "former_record": "#3b82f6",
    "worse_than_record": "#ef4444",
    "matched_optimal": "#6b7280",
    "default": "#e5e7eb",
    "text_white": "#ffffff",
    "text_gray": "#374151",
    "title_main": "#111827",
    "title_sub": "#4b5563",
}

COLS = 10
ROWS = math.ceil(TOTAL_PROBLEMS / COLS)
CELL_W = 1
GAP = 0.15
HEADER_SPACE = 2.5

# --- Drawing ---
fig_height = ((ROWS + HEADER_SPACE) / COLS) * 8
# Optional: Set facecolor here as well for consistency if displayed in a GUI
fig, ax = plt.subplots(figsize=(8, fig_height), facecolor="white")

ax.set_xlim(-GAP, COLS * (CELL_W + GAP))
ax.set_ylim(-GAP, (ROWS + HEADER_SPACE) * (CELL_W + GAP))
ax.set_aspect("equal")
ax.axis("off")

# 1. Draw Grid
for i in range(1, TOTAL_PROBLEMS + 1):
  col_idx = (i - 1) % COLS
  row_idx = (i - 1) // COLS
  plot_row = ROWS - 1 - row_idx
  x = col_idx * (CELL_W + GAP)
  y = plot_row * (CELL_W + GAP)

  bg = COLORS["default"]
  txt = COLORS["text_gray"]
  # Check which category the index falls into
  if i in CATEGORIES.get("world_record", []):
    bg, txt = COLORS["world_record"], COLORS["text_white"]
  elif i in CATEGORIES.get("former_record", []):
    bg, txt = COLORS["former_record"], COLORS["text_white"]
  elif i in CATEGORIES.get("worse_than_record", []):
    bg, txt = COLORS["worse_than_record"], COLORS["text_white"]
  elif i in CATEGORIES.get("matched_optimal", []):
    bg, txt = COLORS["matched_optimal"], COLORS["text_white"]

  box = patches.FancyBboxPatch(
      (x, y),
      CELL_W,
      CELL_W,
      boxstyle="round,pad=0,rounding_size=0.2",
      fc=bg,
      ec="none",
  )
  ax.add_patch(box)
  ax.text(
      x + CELL_W / 2,
      y + CELL_W / 2,
      str(i),
      ha="center",
      va="center",
      color=txt,
      fontsize=11,
      fontfamily="sans-serif",
      fontweight="bold",
  )

grid_top_y = ROWS * (CELL_W + GAP)
center_x = (COLS * (CELL_W + GAP) - GAP) / 2
total_h = (ROWS + HEADER_SPACE) * (CELL_W + GAP)

# 2. Draw Legend
legend_handles = [
    patches.Patch(color=COLORS["world_record"], label="New result"),
    patches.Patch(
        color=COLORS["former_record"],
        label="Former new result, got improved upon",
    ),
    patches.Patch(
        color=COLORS["worse_than_record"], label="Worse than literature bound"
    ),
    patches.Patch(
        color=COLORS["matched_optimal"], label="Matched known optimal bound"
    ),
    patches.Patch(
        color=COLORS["default"], label="Matched literature bound / N/A"
    ),
]

leg = ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, grid_top_y / total_h + 0.01),
    ncol=2,
    frameon=False,
    fontsize=10,
    handlelength=1.2,
    borderaxespad=0.0,
)

# 3. Draw Title & Subtitle
title_y = (ROWS + HEADER_SPACE - 0.7) * (CELL_W + GAP)
subtitle_y = (ROWS + HEADER_SPACE - 1.4) * (CELL_W + GAP)

ax.text(
    center_x,
    title_y,
    "New result distribution",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
    color=COLORS["title_main"],
    fontfamily="sans-serif",
)
ax.text(
    center_x,
    subtitle_y,
    "Visualization of results across 67 problems.",
    ha="center",
    va="center",
    fontsize=14,
    color=COLORS["title_sub"],
    fontfamily="sans-serif",
)

plt.tight_layout()

# Added facecolor='white' to ensure background is not transparent
plt.savefig("status_grid.png", dpi=150, bbox_inches="tight", facecolor="white")
