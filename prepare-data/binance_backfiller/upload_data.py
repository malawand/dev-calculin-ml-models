import os
import sys
import json
from collections import defaultdict
from datetime import datetime, timezone
import clickhouse_connect

# --- Configuration ---
CLICKHOUSE_HOST = '10.1.19.249'
CLICKHOUSE_PORT = 8124
CLICKHOUSE_USER = 'exporter'
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD")

DATABASE = "binance_ww"
AUDIT_TABLE = f"{DATABASE}.data_gaps_audit"
AUDIT_STATUS_AFTER_UPLOAD = "resolved_from_1s_klines"


def parse_iso_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def convert_datetime_columns(data_points: list[list], insert_columns: list[str]) -> None:
    datetime_columns = {"event_time", "open_time", "close_time"}
    for row in data_points:
        for col_name in datetime_columns:
            if col_name not in insert_columns:
                continue
            idx = insert_columns.index(col_name)
            if row[idx] is not None:
                row[idx] = parse_iso_utc(row[idx])


def main():
    # 1. Load generated ticker data
    try:
        with open("ready_for_clickhouse.json", "r") as f:
            payload = json.load(f)
    except FileNotFoundError:
        print("❌ 'ready_for_clickhouse.json' not found. Run recreate_24h_ticker_for_gap.py first!")
        return

    if not payload:
        print("⚠️ No data points to upload. Ending workflow.")
        return

    # 2. Group payload rows by symbol for dedicated table routing
    grouped_by_symbol = defaultdict(list)
    for row in payload:
        symbol = row.get("symbol")
        if not symbol:
            print("⚠️ Skipping row missing 'symbol' field.")
            continue
        grouped_by_symbol[symbol].append(row)

    # 3. Load pending_gaps with symbol & start_str for exact audit reconciliation
    gaps_resolved = []
    try:
        with open("pending_gaps.json", "r") as f:
            pending_gaps = json.load(f)
            gaps_resolved = [
                (gap["symbol"], gap["start_str"])
                for gap in pending_gaps
                if "symbol" in gap and "start_str" in gap
            ]
    except FileNotFoundError:
        print("⚠️ 'pending_gaps.json' not found. Audit table will not be updated.")

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )

    # 4. Insert data into each symbol's dedicated table
    for symbol, rows in grouped_by_symbol.items():
        table_name = f"{DATABASE}.raw_ticks_{symbol}_binance"
        insert_columns = list(rows[0].keys())
        data_points = [list(r.values()) for r in rows]

        convert_datetime_columns(data_points, insert_columns)

        print(f"\n🚀 Inserting {len(data_points)} recovered rows into {table_name}...")
        client.insert(
            table_name,
            data_points,
            column_names=insert_columns,
        )
        print(f"✅ Successfully inserted rows into {table_name}.")

    # 5. Update audit table matching both symbol AND gap_start
    print("\n📝 Updating audit table statuses...")
    for symbol, gap_str in gaps_resolved:
        dt = parse_iso_utc(gap_str)
        ch_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        mutation = f"""
            ALTER TABLE {AUDIT_TABLE}
            UPDATE status = '{AUDIT_STATUS_AFTER_UPLOAD}'
            WHERE symbol = '{symbol}' AND gap_start = '{ch_str}'
            SETTINGS mutations_sync = 1
        """
        client.command(mutation)
        print(f"✅ Marked [{symbol}] gap starting at {ch_str} as '{AUDIT_STATUS_AFTER_UPLOAD}'.")

    print("\n🎉 Workflow complete! Data uploaded and audit table updated.")


if __name__ == "__main__":
    main()