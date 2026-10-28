import requests
import time
import sys
from datetime import datetime, timezone
import clickhouse_connect
import json
import os

# --- Configuration ---
CLICKHOUSE_HOST = '10.1.19.249'
CLICKHOUSE_PORT = 8124
CLICKHOUSE_USER = 'exporter'
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD")
BATCH_SIZE = 20 

def main():
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )

    # 1. Fetch the 'symbol' column alongside the gap timestamps
    query = f"""
        SELECT symbol, gap_start, gap_end, gap_duration_seconds
        FROM binance_ww.data_gaps_audit
        WHERE status = 'upstream_missing_unrecoverable' OR status = 'requires_backfill'
        ORDER BY gap_duration_seconds DESC
        LIMIT {BATCH_SIZE}
    """

    gaps = client.query(query).result_rows

    if not gaps:
        print("🎉 No gaps require backfilling right now!")
        return

    gaps_to_save = []

    # 2. Unpack the dynamic symbol from the database row
    for symbol, start_dt, end_dt, duration in gaps:
        # Force UTC
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        scrape_delay_seconds = 20
        
        print(f"✅ Processing gap for [{symbol}] from {start_dt} to {end_dt}. Applying {scrape_delay_seconds}s offset.")

        gaps_to_save.append(
            {
                "symbol": symbol,  # 3. Use the dynamic symbol here
                "start_str": start_dt.isoformat(),
                "end_str": end_dt.isoformat(),
                "scrape_delay_seconds": scrape_delay_seconds, 
                "duration": duration,
            }
        )

    with open("pending_gaps.json", "w") as f:
        json.dump(gaps_to_save, f, indent=4)

    print(f"✅ Saved {len(gaps_to_save)} gaps to 'pending_gaps.json'.")

if __name__ == "__main__":
    main()