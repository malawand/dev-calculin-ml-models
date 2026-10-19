import os
import logging
import clickhouse_connect

# --- Configuration ---
CLICKHOUSE_HOST = '10.1.19.249'
CLICKHOUSE_PORT = 8124
CLICKHOUSE_USER = 'exporter'
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD")

DATABASE = "binance_ww"
AUDIT_TABLE = "binance_ww.data_gaps_audit"
LOOKBACK_DAYS = 7
GAP_THRESHOLD_SECONDS = 15

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_symbol_tables(client: clickhouse_connect.driver.Client) -> list[tuple[str, str]]:
    """
    Discovers all tables matching 'raw_ticks_%_binance' 
    and extracts the uppercase symbol (e.g. BTCUSDT).
    """
    query = f"""
        SELECT name 
        FROM system.tables 
        WHERE database = '{DATABASE}' 
          AND name LIKE 'raw_ticks_%_binance'
    """
    try:
        result = client.query(query).result_rows
        
        tables_to_process = []
        for row in result:
            table_name = row[0]
            # Slices off 'raw_ticks_' (10 chars) and '_binance' (8 chars)
            symbol_str = table_name[10:-8].upper()
            full_table_path = f"{DATABASE}.{table_name}"
            
            tables_to_process.append((symbol_str, full_table_path))
            
        return tables_to_process
    except Exception as e:
        logger.error(f"Failed to fetch tables from system metadata: {e}")
        return []

def audit_symbol_gaps(client: clickhouse_connect.driver.Client, symbol: str, raw_table: str) -> None:
    """Runs the gap detection against a symbol's dedicated raw table."""
    
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
                FROM {raw_table}
                WHERE event_time >= now() - INTERVAL {LOOKBACK_DAYS} DAY
                  AND symbol = %(symbol)s
                  AND close_time IS NOT NULL 
                  AND last_price > 0
            )
            WHERE next_time IS NOT NULL
              AND gap_duration_seconds > {GAP_THRESHOLD_SECONDS}
        ) AS new_gaps
        LEFT ANTI JOIN {AUDIT_TABLE} a
          ON new_gaps.symbol = a.symbol AND new_gaps.gap_start = a.gap_start
    """
    
    try:
        result = client.query(select_query, parameters={"symbol": symbol}).result_rows
        
        if not result:
            logger.info(f"[{symbol}] Healthy: No new gaps > {GAP_THRESHOLD_SECONDS}s found.")
        else:
            gap_count = len(result)
            logger.warning(f"[{symbol}] Alert: Found {gap_count} new gaps! Inserting to audit table...")
            
            insert_query = f"""
                INSERT INTO {AUDIT_TABLE} 
                (symbol, cluster, job, instance, gap_start, gap_end, gap_duration_seconds, status)
                {select_query}
            """
            client.command(insert_query, parameters={"symbol": symbol})
            logger.info(f"[{symbol}] Success: {gap_count} gaps logged to {AUDIT_TABLE}.")
            
    except Exception as e:
        logger.error(f"[{symbol}] Error querying table {raw_table}: {e}")

def main():
    logger.info("🔍 Starting Multi-Table Gap Audit Pipeline...")

    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
        )
    except Exception as e:
        logger.error(f"FATAL: Failed to connect to ClickHouse. Error: {e}")
        return

    logger.info("Discovering target tables from ClickHouse metadata...")
    active_targets = get_symbol_tables(client)
    
    if not active_targets:
        logger.warning(f"No tables matching 'raw_ticks_%_binance' found in {DATABASE}. Exiting.")
        return
        
    logger.info(f"Found {len(active_targets)} target tables.")
    logger.info("=" * 60)

    for i, (symbol, table_path) in enumerate(active_targets, 1):
        logger.info(f"Processing ({i}/{len(active_targets)}): {symbol} -> {table_path}")
        audit_symbol_gaps(client, symbol, table_path)
        
    logger.info("=" * 60)
    logger.info("✅ Multi-Table Gap Audit Pipeline Complete.")

if __name__ == "__main__":
    main()