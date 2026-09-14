#!/usr/bin/env python3
"""
Production Extractor: Cortex -> Local Parquet -> ClickHouse
Handles strict schemas, memory-safe chunking, specific Binance DDL types,
and streaming database ingestion. Fully hardened against infrastructure drops.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import clickhouse_connect
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import tempfile

# ---------------------------------------------------------------------------
# 1. CONFIGURATION & SCHEMA DEFINITION
# ---------------------------------------------------------------------------
MAX_SAMPLES_PER_REQUEST = 10_000
MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0

# Map Cortex PromQL metrics to ClickHouse DDL columns
METRIC_MAPPING = {
    "crypto_last_price": "last_price",
    "crypto_bid_price": "bid_price",
    "crypto_ask_price": "ask_price",
    "crypto_bid_qty": "bid_qty",
    "crypto_ask_qty": "ask_qty",
    "crypto_last_qty": "last_qty",
    "crypto_open_price": "open_price",
    "crypto_high_price": "high_price",
    "crypto_low_price": "low_price",
    "crypto_prev_close_price": "prev_close_price",
    "crypto_weighted_average_price": "weighted_avg_price",
    "crypto_price_change": "price_change",
    "crypto_price_change_percent": "price_change_percent",
    "crypto_volume": "volume",
    "crypto_quote_volume": "quote_volume",
    "crypto_count": "trade_count",
    "crypto_first_id": "first_trade_id",
    "crypto_last_id": "last_trade_id",
    "crypto_open_time": "open_time",
    "crypto_close_time": "close_time"
}

# STRICT SCHEMA: perfectly matches the raw_ticks_binance DDL types
PA_FIELDS = [
    pa.field("symbol", pa.string()),
    pa.field("event_time", pa.timestamp('ms', tz='UTC')),
    pa.field("last_price", pa.float64()),
    pa.field("bid_price", pa.float64()),
    pa.field("ask_price", pa.float64()),
    pa.field("bid_qty", pa.float64()),
    pa.field("ask_qty", pa.float64()),
    pa.field("last_qty", pa.float64()),
    pa.field("open_price", pa.float64()),
    pa.field("high_price", pa.float64()),
    pa.field("low_price", pa.float64()),
    pa.field("prev_close_price", pa.float64()),
    pa.field("weighted_avg_price", pa.float64()),
    pa.field("price_change", pa.float64()),
    pa.field("price_change_percent", pa.float64()),
    pa.field("volume", pa.float64()),
    pa.field("quote_volume", pa.float64()),
    pa.field("trade_count", pa.uint64()),
    pa.field("first_trade_id", pa.uint64()),
    pa.field("last_trade_id", pa.uint64()),
    pa.field("open_time", pa.timestamp('ms', tz='UTC')),
    pa.field("close_time", pa.timestamp('ms', tz='UTC')),
    pa.field("cluster", pa.string()),
    pa.field("instance", pa.string()),
    pa.field("job", pa.string())

]
STRICT_SCHEMA = pa.schema(PA_FIELDS)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2. CORE FUNCTIONS
# ---------------------------------------------------------------------------
def _parse_dt(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {value!r}")

def _query_cortex(url: str, query: str, start_ts: int, end_ts: int, step: int) -> List[Dict[str, Any]]:
    """Fetch data from Cortex with exponential backoff."""
    params = {"query": query, "start": start_ts, "end": end_ts, "step": f"{step}s"}
    backoff = INITIAL_BACKOFF_S

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code in {429, 500, 502, 503, 504}:
                logger.warning(f"Cortex HTTP {resp.status_code}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()

            body = resp.json()
            if body.get("status") != "success":
                raise ValueError(f"Cortex error: {body}")

            return body.get("data", {}).get("result", [])

        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                logger.error(f"Failed to query Cortex after {MAX_RETRIES} attempts: {query}")
                raise
            time.sleep(backoff)
            backoff *= 2
    return []

# ---------------------------------------------------------------------------
# 3. EXTRACTION TO DISK (Streaming Parquet)
# ---------------------------------------------------------------------------
def extract_to_parquet(
    url: str, symbol: str, start: datetime, end: datetime, step: int, output_file: str
) -> tuple[int, bool]:
    """Extracts data in chunks, aligns metrics, casts types, and streams to Parquet."""
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    total_seconds = end_ts - start_ts

    chunk_seconds = (MAX_SAMPLES_PER_REQUEST - 1) * step
    n_chunks = max(1, math.ceil(total_seconds / chunk_seconds))

    logger.info(f"Target: {symbol} | Total time: {total_seconds}s | Step: {step}s")
    logger.info(f"Splitting into {n_chunks} safe chunks to prevent Cortex OOM.")

    total_rows_written = 0
    writer = None
    extraction_aborted = False  # Track if we stopped early due to an error

    try:
        chunk_start = start_ts
        for i in range(n_chunks):
            chunk_end = min(chunk_start + chunk_seconds, end_ts)
            logger.info(f"--- Processing Chunk {i+1}/{n_chunks} ({datetime.fromtimestamp(chunk_start, timezone.utc)} to {datetime.fromtimestamp(chunk_end, timezone.utc)}) ---")

            try:
                chunk_series = {}

                # Fetch every metric for this specific chunk
                for prom_metric in METRIC_MAPPING.keys():
                    query = f'{prom_metric}{{symbol="{symbol}"}}'
                    results = _query_cortex(url, query, chunk_start, chunk_end, step)

                    metric_parts = []

                    if results:
                        for series in results:
                            values = series.get("values", [])

                            if not values:
                                continue

                            idx = pd.to_datetime([float(v[0]) for v in values], unit="s", utc=True)
                            vals = [float(v[1]) for v in values]

                            metric_parts.append(pd.Series(vals, index=idx))
                            # -------------------------

                    if metric_parts:
                        metric_series = pd.concat(metric_parts)

                        # If two series have the same timestamp, keep the latest one.
                        # This protects against overlapping pods/instances.
                        metric_series = metric_series[~metric_series.index.duplicated(keep="last")]
                        metric_series = metric_series.sort_index()

                        chunk_series[prom_metric] = metric_series
                    else:
                        chunk_series[prom_metric] = pd.Series(dtype="float64")

                # Align all metrics on exact timestamps
                if not chunk_series:
                    logger.error(f"FATAL: No data found for any metric in chunk {i+1}. Cortex might be silently dropping data.")
                    raise ValueError(f"Empty Cortex response for {symbol} between {chunk_start} and {chunk_end}. Failing pod to prevent permanent data gap.")

                chunk_df = pd.DataFrame(chunk_series)
                if not chunk_df.empty:
                    diffs = chunk_df.index.to_series().sort_values().diff().dt.total_seconds().dropna()
                    big_gaps = diffs[diffs > step]

                    if not big_gaps.empty:
                        logger.warning(
                            "Chunk contains timestamp gaps | symbol=%s | start=%s | end=%s | gap_count=%s | largest_gap=%s",
                            symbol,
                            datetime.fromtimestamp(chunk_start, timezone.utc),
                            datetime.fromtimestamp(chunk_end, timezone.utc),
                            len(big_gaps),
                            big_gaps.max(),
                        )

                # 1. Rename the index to match the DDL time column
                chunk_df.index.name = "event_time"
                chunk_df = chunk_df.reset_index()

                # 2. Inject the symbol column
                chunk_df.insert(0, 'symbol', symbol)

                # --- DYNAMIC INGESTION LINEAGE ---
                # HOSTNAME is automatically injected by Kubernetes for every Pod
                pod_name = os.getenv('HOSTNAME', 'dev-pod')
                cluster_name = os.getenv('CLUSTER_NAME', 'dev-cluster')
                job_name = os.getenv('JOB_NAME', 'binance-extraction')

                chunk_df.insert(1, 'cluster', cluster_name)
                chunk_df.insert(2, 'instance', pod_name)
                chunk_df.insert(3, 'job', job_name)
                # ---------------------------------

                # 3. Rename Cortex columns to ClickHouse DDL columns
                chunk_df = chunk_df.rename(columns=METRIC_MAPPING)

                # 4. Fill missing columns and enforce casting
                for prom_metric, ddl_col in METRIC_MAPPING.items():
                    if ddl_col not in chunk_df.columns:
                        chunk_df[ddl_col] = float('nan')

                    # Cast UInt64 columns (replacing NaNs with 0 to prevent float casting errors)
                    if ddl_col in ["trade_count", "first_trade_id", "last_trade_id"]:
                        chunk_df[ddl_col] = chunk_df[ddl_col].fillna(0).astype('uint64')

                    # Cast secondary timestamps
                    elif ddl_col in ["open_time", "close_time"]:
                        chunk_df[ddl_col] = pd.to_datetime(chunk_df[ddl_col], unit='ms', utc=True)

                # Ensure columns are in the exact order as PyArrow schema
                ordered_columns = [f.name for f in PA_FIELDS]
                chunk_df = chunk_df[ordered_columns]

                rows_in_chunk = len(chunk_df)
                if rows_in_chunk > 0:
                    # Convert to PyArrow table with strict DDL-matching schema
                    table = pa.Table.from_pandas(chunk_df, schema=STRICT_SCHEMA, preserve_index=False)

                    if writer is None:
                        writer = pq.ParquetWriter(output_file, STRICT_SCHEMA, compression='snappy')

                    writer.write_table(table)
                    total_rows_written += rows_in_chunk
                    logger.info(f"Chunk {i+1} complete. Wrote {rows_in_chunk} aligned rows to Parquet.")
                else:
                    logger.info(f"Chunk {i+1} yielded 0 rows after alignment.")

                chunk_start = chunk_end
                time.sleep(0.1)

            # Catch the chunk failure and break the loop gracefully
            except Exception as e:
                logger.error(f"Extraction failed abruptly at Chunk {i+1}. Error: {e}")
                logger.warning("Halting extraction early to save previously successful chunks...")
                extraction_aborted = True
                break  # Stops the for-loop, jumps down to the finally block

    finally:
        if writer:
            writer.close()

    logger.info(f"Extraction Phase Complete. Total rows staged on disk: {total_rows_written}")

    # Return both the row count and the boolean flag
    return total_rows_written, extraction_aborted

# ---------------------------------------------------------------------------
# 4. LOAD TO CLICKHOUSE (Streaming from Disk)
# ---------------------------------------------------------------------------
def load_to_clickhouse(
    ch_host: str, ch_port: int, ch_user: str, ch_pass: str,
    ch_table: str, file_path: str
) -> None:
    """Streams the local Parquet file into ClickHouse natively via PyArrow."""
    if not os.path.exists(file_path):
        logger.error(f"Parquet file {file_path} not found. Skipping load phase.")
        return

    # --- STRUCTURAL INTEGRITY CHECK ---
    try:
        parquet_file = pq.ParquetFile(file_path)
        total_batches = parquet_file.num_row_groups
    except Exception as e:
        logger.error(f"FATAL: Staged Parquet file is corrupted or unreadable! Storage failure suspected. Error: {e}")
        raise IOError(f"Aborting insertion phase to prevent loading corrupt data from {file_path}")

    # --- EXPONENTIAL BACKOFF RETRY FOR DATABASE LOADING ---
    inserted_rows = 0
    ch_backoff = 2.0
    max_ch_retries = 3

    # Track exactly which batch we are on so we don't re-upload duplicates
    current_batch = 0

    for attempt in range(1, max_ch_retries + 1):
        try:
            logger.info(f"Connecting to ClickHouse at {ch_host}:{ch_port} (Attempt {attempt}/{max_ch_retries})...")
            client = clickhouse_connect.get_client(
                host=ch_host, port=ch_port, username=ch_user, password=ch_pass
            )

            if attempt == 1:
                logger.info(f"Parquet file verified. Contains {total_batches} row groups. Streaming to {ch_table}...")
            else:
                logger.info(f"Resuming upload from batch {current_batch+1}/{total_batches}...")

            # Start the loop from 'current_batch' instead of 0
            for i in range(current_batch, total_batches):
                batch_table = parquet_file.read_row_group(i)
                client.insert_arrow(ch_table, batch_table)
                inserted_rows += batch_table.num_rows
                logger.info(f"Inserted batch {i+1}/{total_batches} ({batch_table.num_rows} rows).")

                # Mark this batch as successfully completed
                current_batch += 1

            logger.info(f"Load Phase Complete. Successfully inserted {inserted_rows} rows into ClickHouse.")
            break  # Success! Break out of the retry loop.

        except Exception as e:
            logger.warning(f"ClickHouse insertion failed on attempt {attempt}/{max_ch_retries}. Error: {e}")
            if attempt == max_ch_retries:
                logger.error("FATAL: Maximum ClickHouse insertion retries exhausted.")
                raise

            logger.info(f"Retrying database operations in {ch_backoff}s...")
            time.sleep(ch_backoff)
            ch_backoff *= 2

# ---------------------------------------------------------------------------
# 5. ORCHESTRATION
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Production Cortex to ClickHouse Pipeline")
    parser.add_argument("--symbol", required=True, help="Binance symbol (e.g., BTCUSDT)")

    # Set required=False so the script can auto-detect dates when these are missing
    parser.add_argument("--start", required=False, help="Start date (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end", required=False, help="End date (YYYY-MM-DD HH:MM:SS)")

    parser.add_argument("--step", type=int, default=3600, help="Resolution step in seconds")
    parser.add_argument("--cortex-url", required=True, help="Cortex query_range endpoint")

    # ClickHouse Args
    parser.add_argument("--ch-host", required=True)
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--ch-user", default="default")
    parser.add_argument("--ch-pass", default="")
    parser.add_argument("--ch-table", default="binance_ww.raw_ticks_binance")

    args = parser.parse_args()

    # PHASE 1: Dynamic start logic (Queries ClickHouse max event_time if --start is missing)
    if args.start:
        start_str = args.start
    else:
        logger.info(f"No --start provided. Querying ClickHouse for latest {args.symbol} event_time...")
        ch_client = clickhouse_connect.get_client(
            host=args.ch_host, port=args.ch_port, username=args.ch_user, password=args.ch_pass
        )
        max_ts = ch_client.command(f"SELECT max(event_time) FROM {args.ch_table} WHERE symbol = '{args.symbol}'")

        if not max_ts or str(max_ts).startswith('1970'):
            logger.error("Table is empty or returns epoch. You MUST provide a --start argument for the first run.")
            exit(1)

        start_str = pd.to_datetime(max_ts).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Auto-detected start time from ClickHouse: {start_str}")

    # PHASE 2: Dynamic end logic (Default to current UTC time if --end is missing)
    if args.end:
        end_str = args.end
    else:
        end_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"No --end provided. Defaulting to current time: {end_str}")

    start_dt = _parse_dt(start_str)
    end_dt = _parse_dt(end_str)

    # PHASE 3: The 6-Month (180 days) window cap safety block
    MAX_WINDOW = timedelta(days=180)
    if (end_dt - start_dt) > MAX_WINDOW:
        logger.warning(f"The data gap ({end_dt - start_dt}) is wider than 6 months.")
        end_dt = start_dt + MAX_WINDOW
        logger.warning(f"Capping this run's end target to: {end_dt}")

    temp_dir = tempfile.gettempdir()
    temp_parquet = f"{temp_dir}/{args.symbol}_ticks_{int(start_dt.timestamp())}.parquet"
    logger.info("=== Starting Pipeline ===")
    logger.info(f"Symbol: {args.symbol} | Range: {start_dt} -> {end_dt}")

    # 1. Extract & Stage
    # Unpack both the rows count and the aborted flag
    rows_extracted, extraction_aborted = extract_to_parquet(
        url=args.cortex_url,
        symbol=args.symbol,
        start=start_dt,
        end=end_dt,
        step=args.step,
        output_file=temp_parquet
    )

    # 2. Validate and Load
    if rows_extracted > 0:
        from validator import validate_crypto_data

        logger.info("Running pre-load validation...")

        is_valid = validate_crypto_data(
            file_path=temp_parquet,
            step_seconds=args.step,
            ch_host=args.ch_host,
            ch_port=args.ch_port,
            ch_user=args.ch_user,
            ch_pass=args.ch_pass,
            ch_table=args.ch_table
        )

        if not is_valid:
            logger.error("Validation failed. Aborting pipeline before loading bad data.")
            return

        # 3. Load
        load_to_clickhouse(
            ch_host=args.ch_host,
            ch_port=args.ch_port,
            ch_user=args.ch_user,
            ch_pass=args.ch_pass,
            ch_table=args.ch_table,
            file_path=temp_parquet
        )

        # 4. Determine final exit state based on the abortion flag
        if extraction_aborted:
            logger.error("FATAL: Pipeline safely loaded partial data but aborted mid-extraction. Failing pod to trigger Kubernetes restart.")
            sys.exit(1)
        else:
            logger.info("=== Pipeline Completed Successfully ===")

    else:
        logger.error("FATAL: Pipeline aborted before load phase: No data extracted.")
        sys.exit(1)

if __name__ == "__main__":
    main()
