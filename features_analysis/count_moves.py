#!/usr/bin/env python3
"""
count_moves.py
==============

For each combination, count how many times the price went UP, DOWN, or was
FLAT over the next N hours.

    python count_moves.py cortex_features_with_prices.csv
    python count_moves.py cortex_features_with_prices.csv --normalize
    python count_moves.py cortex_features_with_prices.csv --hours 4 --threshold 0.5
"""

import argparse
import sys
from itertools import product

import pandas as pd
import numpy as np  # Added for log normalization

FEATURES = ["7d_derivative", "7d_derivative_prime", "poly_transition(_weighted24h48h7d)"]

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--features", nargs="+", default=FEATURES)
p.add_argument("--hours", type=int, default=24, help="lookahead in hours")
p.add_argument("--threshold", type=float, default=0.1,
               help="moves smaller than this %% count as FLAT")
p.add_argument("--time-col", default="time_bucket")
p.add_argument("--price-col", default="close_price")
p.add_argument("--normalize", action="store_true", 
               help="Use Logarithmic Returns to mathematically normalize extreme price variations")
a = p.parse_args()

F = a.features

df = pd.read_csv(a.csv)
missing = [c for c in F + [a.time_col, a.price_col] if c not in df.columns]
if missing:
    sys.exit(f"ERROR: columns not found: {missing}\nAvailable: {list(df.columns)}")

# --- prep: sort by time, one row per hour ---
df[a.time_col] = pd.to_datetime(df[a.time_col], utc=True, errors="coerce")
df = (df.dropna(subset=[a.time_col]).sort_values(a.time_col)
        .drop_duplicates(subset=[a.time_col], keep="last").set_index(a.time_col))
for c in F + [a.price_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC"))

# --- the state label, e.g. "001" ---
ok = df[F].notna().all(axis=1)
df["state"] = None
df.loc[ok, "state"] = df.loc[ok, F].astype(int).astype(str).agg("".join, axis=1)

# --- price change over the next N hours ---
if a.normalize:
    # Log Returns: Normalizes massive swings symmetrically
    df["change"] = np.log(df[a.price_col].shift(-a.hours) / df[a.price_col]) * 100
else:
    # Standard percentage change
    df["change"] = (df[a.price_col].shift(-a.hours) / df[a.price_col] - 1) * 100

# --- classify ---
df["result"] = None
df.loc[df["change"] > a.threshold, "result"] = "UP"
df.loc[df["change"] < -a.threshold, "result"] = "DOWN"
df.loc[df["change"].abs() <= a.threshold, "result"] = "FLAT"

rows = df.dropna(subset=["state", "result"])

# --- output ---
mode_str = "LOG NORMALIZED" if a.normalize else "STANDARD PERCENTAGE"
print(f"\nLookahead: {a.hours} hours      FLAT threshold: +/-{a.threshold}%      Mode: {mode_str}")
print("Features:  " + "  ".join(f"digit{i+1}={f}" for i, f in enumerate(F)))
print()
print(f"{'state':<8}{'UP':>8}{'DOWN':>8}{'FLAT':>8}{'TOTAL':>9}"
      f"{'UP%':>8}{'DOWN%':>8}{'FLAT%':>8}")
print("-" * 63)

for s in ["".join(t) for t in product("01", repeat=len(F))]:
    g = rows[rows["state"] == s]
    n = len(g)
    if n == 0:
        print(f"{s:<8}{'-':>8}{'-':>8}{'-':>8}{0:>9}")
        continue
    up = (g["result"] == "UP").sum()
    dn = (g["result"] == "DOWN").sum()
    fl = (g["result"] == "FLAT").sum()
    print(f"{s:<8}{up:>8,}{dn:>8,}{fl:>8,}{n:>9,}"
          f"{100*up/n:>7.1f}%{100*dn/n:>7.1f}%{100*fl/n:>7.1f}%")

print("-" * 63)
n = len(rows)
up = (rows["result"] == "UP").sum()
dn = (rows["result"] == "DOWN").sum()
fl = (rows["result"] == "FLAT").sum()
print(f"{'ALL':<8}{up:>8,}{dn:>8,}{fl:>8,}{n:>9,}"
      f"{100*up/n:>7.1f}%{100*dn/n:>7.1f}%{100*fl/n:>7.1f}%")
print()