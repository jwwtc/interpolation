#!/usr/bin/env python3
"""
interpolate_cellwise.py ─ per-voxel cleaning, no-shift
=====================================================

Workflow
--------
1.  Read raw melt-pool history (x, y, z, tm, tl, cr) within a
    1 × 1 × 0.4 mm domain.
2.  Replicate every coarse 20 µm event to all 64 (4×4×4) fine
    5 µm voxels of its parent cube.
3.  **For each fine voxel** keep at most
       MAX_LAYERS (earliest layers) × MAX_EVENTS (earliest hits per layer)
    where layers are separated by DWELL_DT seconds.
4.  Write the cleaned 5 µm file; report tm≤tl overlaps.

Change log
----------
* *clean_voxel* now keeps the **earliest** layers so each node accounts for
  its own layer plus the two layers above.
"""

from pathlib import Path
import time, collections
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── user settings ──────────────────────────────────────────────────────────
GEOM_DIR    = Path("data/1x1x04")
INPUT_FILE  = GEOM_DIR / "melt-pool-history.txt"
OUTPUT_FILE = GEOM_DIR / "melt_pool_history_5um_cellwise.txt"

COARSE_DX  = 20e-6          # 20 µm parent cell
FINE_DX    =  5e-6          # 5 µm voxel
DOM_X      = (0.0, 1.0e-3)  # m
DOM_Y      = (0.0, 1.0e-3)
DOM_Z      = (0.0, 4.0e-4)

MAX_LAYERS = 3              # per fine voxel
MAX_EVENTS = 2              # per layer
DWELL_DT   = 5.0            # s  → layer split gap
COORD_PREC = 10             # decimals for coordinates
# ───────────────────────────────────────────────────────────────────────────


def load_history(path: Path) -> pd.DataFrame:
    with path.open() as f:
        has_header = f.readline().lower().startswith("x")
    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=["x", "y", "z", "tm", "tl", "cr"] if not has_header else None,
        dtype=float,
    )
    mask = (
        df.x.between(*DOM_X)
        & df.y.between(*DOM_Y)
        & df.z.between(*DOM_Z)
        & (df.tl > df.tm)
    )
    df = df[mask]
    print(f"Loaded {len(df):,} coarse events")
    return df


def make_layers(times: np.ndarray):
    """Yield index lists whose first–last gap ≤ DWELL_DT (ascending order)."""
    start = 0
    n = len(times)
    while start < n:
        end = start
        while end + 1 < n and times[end + 1] - times[start] <= DWELL_DT:
            end += 1
        yield list(range(start, end + 1))
        start = end + 1


def clean_voxel(events: list) -> list:
    """
    Keep up to MAX_LAYERS × MAX_EVENTS events for this 5 µm voxel.

    Strategy:
      • build ALL layers (ascending time),
      • take the **earliest** MAX_LAYERS layers,
      • within each layer keep the earliest MAX_EVENTS hits.
    """
    events.sort(key=lambda e: e["tm"])  # ascending tm
    times = np.fromiter((e["tm"] for e in events), float)

    layers = list(make_layers(times))            # earliest → latest
    selected = layers[:MAX_LAYERS] or []

    kept = []
    for idxs in selected:
        kept.extend(events[i] for i in idxs[:MAX_EVENTS])

    kept.sort(key=lambda e: e["tm"])
    return kept


def main():
    t0 = time.time()
    print(f"=== per-voxel cleaning  (DWELL_DT = {DWELL_DT} s) ===")
    df = load_history(INPUT_FILE)

    vox = collections.defaultdict(list)  # (fx,fy,fz) → list(events)

    # ── replicate ─────────────────────────────────────────────────────────
    print("Replicating coarse events to 5 µm voxels …")
    for r in tqdm(df.itertuples(index=False), total=len(df)):
        ix = int((r.x - DOM_X[0]) // COARSE_DX)
        iy = int((r.y - DOM_Y[0]) // COARSE_DX)
        iz = int((r.z - DOM_Z[0]) // COARSE_DX)

        x0 = DOM_X[0] + ix * COARSE_DX
        y0 = DOM_Y[0] + iy * COARSE_DX
        z0 = DOM_Z[0] + iz * COARSE_DX

        for sx in range(4):
            fx = round(x0 + sx * FINE_DX, COORD_PREC)
            for sy in range(4):
                fy = round(y0 + sy * FINE_DX, COORD_PREC)
                for sz in range(4):
                    fz = round(z0 - sz * FINE_DX, COORD_PREC)
                    if fz < DOM_Z[0] - 1e-12:
                        continue
                    vox[(fx, fy, fz)].append(
                        dict(x=fx, y=fy, z=fz, tm=r.tm, tl=r.tl, cr=r.cr)
                    )

    # ── per-voxel cleaning ───────────────────────────────────────────────
    print("Applying layer/event limits per voxel …")
    fine_rows, dropped = [], 0
    for lst in tqdm(vox.values(), total=len(vox)):
        before = len(lst)
        fine_rows.extend(clean_voxel(lst))
        dropped += before - len(clean_voxel(lst))

    print(f"Total events after cleaning: {len(fine_rows):,}")
    print(f"Events dropped by limits:    {dropped:,}")

    # ── write CSV ────────────────────────────────────────────────────────
    cols = ["x", "y", "z", "tm", "tl", "cr"]
    fine_df = (
        pd.DataFrame(fine_rows, columns=cols)
        .sort_values(["z", "x", "y", "tm"])
    )
    fine_df.to_csv(OUTPUT_FILE, index=False, header=False, float_format="%.10f")
    print(f"Written {OUTPUT_FILE}")

    # ── overlap statistic ────────────────────────────────────────────────
    viol = pairs = 0
    for _, g in fine_df.groupby(["x", "y", "z"], sort=False):
        arr = g[["tm", "tl"]].to_numpy()
        if arr.shape[0] < 2:
            continue
        tm, tl = arr[:, 0], arr[:, 1]
        pairs += tm.size - 1
        viol += np.sum(tm[1:] <= tl[:-1])
    pct = 100 * viol / pairs if pairs else 0.0

    print(f"Voxel pairs checked: {pairs:,}")
    print(f"Overlaps (tm≤tl):   {viol:,}  →  {pct:.3f}%")
    print(f"Elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()