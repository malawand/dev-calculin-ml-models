
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




TARGET_TABLE = "binance_ww.raw_ticks_binance"
AUDIT_TABLE = "binance_ww.data_gaps_audit"
AUDIT_STATUS_AFTER_UPLOAD = "resolved_from_1s_klines"


def parse_iso_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def convert_datetime_columns(data_points, insert_columns):
    datetime_columns = {
        "event_time",
        "open_time",
        "close_time",
    }

    for row in data_points:
        for col_name in datetime_columns:
            if col_name not in insert_columns:
                continue

            idx = insert_columns.index(col_name)

            if row[idx] is not None:
                row[idx] = parse_iso_utc(row[idx])


def main():

    # 1. Load the data points
    try:
        with open("ready_for_clickhouse.json", "r") as f:
            payload = json.load(f) # This is a list of dictionaries
    except FileNotFoundError:
        print("❌ 'ready_for_clickhouse.json' not found. Run script #2 first!")
        return

    if not payload:
        print("⚠️ No data points to upload. Ending workflow.")
        return

    # 2. Dynamically extract the columns and convert rows to lists for ClickHouse
    insert_columns = list(payload[0].keys())
    data_points = [list(row.values()) for row in payload]

    # 3. Load pending_gaps to know what to update in the audit table
    gaps_resolved_strs = []
    try:
        with open("pending_gaps.json", "r") as f:
            pending_gaps = json.load(f)
            gaps_resolved_strs = [gap["start_str"] for gap in pending_gaps]
    except FileNotFoundError:
        print("⚠️ 'pending_gaps.json' not found. Audit table will not be updated.")

    print("Columns that will be inserted:")
    print(insert_columns)

    print("\nClickHouse will fill omitted columns with defaults:")
    print(
        [
            "exchange", "bid_price", "ask_price", "bid_qty",
            "ask_qty", "prev_close_price", "weighted_avg_price",
            "price_change", "price_change_percent", "updated_at",
        ]
    )

    # Convert strings to datetime objects
    convert_datetime_columns(data_points, insert_columns)

    print("\nPreview first row after datetime conversion:")
    print(data_points[0])

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )



    print("Inserting rows...")

    client.insert(
        TARGET_TABLE,
        data_points,
        column_names=insert_columns,
    )

    print(f"✅ Inserted {len(data_points)} recovered rows into {TARGET_TABLE}.")

    # Update the audit table for every gap we just fixed
    for gap_str in gaps_resolved_strs:
        dt = parse_iso_utc(gap_str)
        ch_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        mutation = f"""
            ALTER TABLE {AUDIT_TABLE}
            UPDATE status = '{AUDIT_STATUS_AFTER_UPLOAD}'
            WHERE gap_start = '{ch_str}'
            SETTINGS mutations_sync = 1
        """

        client.command(mutation)
        print(f"✅ Marked gap starting at {ch_str} as '{AUDIT_STATUS_AFTER_UPLOAD}'.")

    print("🎉 Workflow complete! Data uploaded and audit table updated.")


if __name__ == "__main__":
    main()
