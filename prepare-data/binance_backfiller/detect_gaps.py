import os
import clickhouse_connect
from datetime import datetime

# --- Configuration ---
CLICKHOUSE_HOST = '10.1.19.249'
CLICKHOUSE_PORT = 8124
CLICKHOUSE_USER = 'exporter'
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD")

RAW_TABLE = "binance_ww.raw_ticks_binance"
AUDIT_TABLE = "binance_ww.data_gaps_audit"
SYMBOL = "BTCUSDT"

def main():
    print("🔍 Starting Weekly Gap Detection...")

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )

    select_query = f"""
        SELECT 
            new_gaps.symbol, 
            new_gaps.cluster,
            new_gaps.job,
            new_gaps.instance,
            new_gaps.gap_start, 
            new_gaps.gap_end, 
            new_gaps.gap_duration_seconds, 
            new_gaps.status
        FROM (
            SELECT
                symbol,
                cluster,
                job,
                instance,
                event_time AS gap_start,
                next_time AS gap_end,
                dateDiff('second', event_time, next_time) AS gap_duration_seconds,
                'requires_backfill' AS status
            FROM (
                SELECT
                    symbol,
                    cluster,
                    job,
                    instance,
                    event_time,
                    any(event_time) OVER (
                        PARTITION BY symbol
                        ORDER BY event_time
                        ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING
                    ) AS next_time
                FROM {RAW_TABLE}
                WHERE event_time >= now() - INTERVAL 7 DAY
                  AND symbol = '{SYMBOL}'
                  AND close_time IS NOT NULL 
                  AND last_price > 0
            )
            WHERE next_time IS NOT NULL
              AND gap_duration_seconds > 15
        ) AS new_gaps
        LEFT ANTI JOIN {AUDIT_TABLE} a
          ON new_gaps.symbol = a.symbol AND new_gaps.gap_start = a.gap_start
    """

    print("📊 Scanning the last 7 days of data for missing rows...")
    
    # Fetch the results to see if we need to insert anything
    result = client.query(select_query).result_rows
    
    if not result:
        print("🎉 No new gaps found in the last 7 days! Database is healthy.")
    else:
        print(f"⚠️ Found {len(result)} new gaps! Inserting them into {AUDIT_TABLE}...")
        
        # Build the INSERT query dynamically using the exact SELECT query
        insert_query = f"""
            INSERT INTO {AUDIT_TABLE} 
            (symbol, cluster, job, instance, gap_start, gap_end, gap_duration_seconds, status)
            {select_query}
        """
        
        # Execute the insert
        client.command(insert_query)
        
        print("✅ New gaps successfully logged to the audit table. Ready for the Backfiller!")

if __name__ == "__main__":
    main()
