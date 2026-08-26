#!/usr/bin/env python3
"""Pull crypto_last_price and crypto_volume from Cortex as far back as they exist."""

from datetime import datetime, timezone

import pandas as pd

from fetch_cortex_metrics import fetch_metric_in_chunks

CORTEX_START = "2023-03-24T00:00:00Z"
OUTPUT_FILE = "btc_last_price.csv"

QUERIES = {
    "close_price": 'max(crypto_last_price{symbol="BTCUSDT"})',
    "volume": 'max(crypto_volume{symbol="BTCUSDT"})',
}


def main() -> None:
    start = int(pd.Timestamp(CORTEX_START, tz="UTC").timestamp())
    end = int(datetime.now(timezone.utc).timestamp())
    print(f"Fetching {CORTEX_START} -> now")

    frames = []
    for name, query in QUERIES.items():
        frame = fetch_metric_in_chunks(name, query, start, end)
        if frame.empty:
            raise SystemExit(f"No data for {name}")
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer")

    merged["time_bucket"] = pd.to_datetime(merged["timestamp"], unit="s", utc=True)
    merged = merged.drop(columns=["timestamp"]).set_index("time_bucket").sort_index()
    grid = pd.date_range(merged.index.min(), merged.index.max(), freq="h", tz="UTC")
    merged = merged.reindex(grid)
    merged["close_price"] = merged["close_price"].ffill(limit=2)
    merged["volume"] = merged["volume"].ffill(limit=2)
    merged = merged.dropna(subset=["close_price"]).reset_index().rename(
        columns={"index": "time_bucket"}
    )
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(merged)} rows to {OUTPUT_FILE}")
    print(f"Range {merged['time_bucket'].min()} -> {merged['time_bucket'].max()}")


if __name__ == "__main__":
    main()
