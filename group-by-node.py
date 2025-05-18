#!/usr/bin/env python3
"""
group-by-node.py  –  reorder melt-pool events so that every (x, y, z) node’s
hits are contiguous and sorted by melting time (tm), keeping ten-decimal
formatting.

Run with NO arguments:

    python3 group-by-node.py

Edit the three path variables below if your files live elsewhere.
"""

from pathlib import Path
import numpy as np

# ── user paths ────────────────────────────────────────────────────────────
GEOM_DIR    = Path("data/1x1x04")
INPUT_FILE  = GEOM_DIR / "melt-pool-history.txt"        # raw history
OUTPUT_FILE = GEOM_DIR / "melt-pool-history_grouped.txt"  # reordered file
# ──────────────────────────────────────────────────────────────────────────

# ── detect header ---------------------------------------------------------
with INPUT_FILE.open() as f:
    first_line = f.readline()
has_header = first_line.lower().startswith("x")
skip = 1 if has_header else 0

# ── load as float64 numpy array ------------------------------------------
print(f"Reading  {INPUT_FILE}  (header present: {has_header})")
raw = np.loadtxt(INPUT_FILE, delimiter=",", dtype=np.float64, skiprows=skip)

# ── sort by x → y → z → tm -----------------------------------------------
order = np.lexsort((raw[:, 3], raw[:, 2], raw[:, 1], raw[:, 0]))
sorted_rows = raw[order]

# ── write with ten-decimal, zero-padded format ---------------------------
fmt = "%.10f,%.10f,%.10f,%.10f,%.10f,%.10f"
print(f"Writing  {OUTPUT_FILE}")
with OUTPUT_FILE.open("w") as out:
    for row in sorted_rows:
        out.write(fmt % tuple(row) + "\n")

print(f"Done – wrote {sorted_rows.shape[0]:,} events.")