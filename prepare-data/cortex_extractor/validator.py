#!/usr/bin/env python3
"""
Advanced Data Validator for Crypto Market Parquet Files.
Checks for time gaps, stale data (flatlines), and ClickHouse schema drift.
"""
import logging
import pandas as pd
import clickhouse_connect

# Set up logging for the validator
logger = logging.getLogger(__name__)

def validate_crypto_data(
    file_path: str,
    step_seconds: int,
    ch_host: str,
    ch_port: int,
    ch_user: str,
    ch_pass: str,
    ch_table: str
) -> bool:
    """
    Validates Parquet data for gaps, staleness, and schema alignment.
    Returns True if safe to load, False if critical schema errors exist.
    """
    logger.info(f"Starting advanced validation for: {file_path}")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to read Parquet file: {e}")
        return False

    if df.empty:
        logger.warning("File is empty. Nothing to validate.")
        return False

    errors = []
    warnings = []

    # Ensure event_time is a datetime object and data is sorted chronologically
    df['event_time'] = pd.to_datetime(df['event_time'], utc=True)
    df = df.sort_values('event_time')

    # ---------------------------------------------------------
    # 1. GAP DETECTION (Logged as Warning)
    # ---------------------------------------------------------
    diffs = df['event_time'].diff().dt.total_seconds().dropna()
    gaps = diffs[diffs > step_seconds]

    if not gaps.empty:
        max_gap = gaps.max()
        warnings.append(
            f"Detected {len(gaps)} missing time gaps! Largest gap: {max_gap}s "
            f"(Expected step: {step_seconds}s)."
        )

    # ---------------------------------------------------------
    # 2. STALE DATA DETECTION (Logged as Warning)
    # ---------------------------------------------------------
    STALE_THRESHOLD_SEC = 600  # 10 minutes

    if 'last_price' in df.columns:
        price_changed = df['last_price'] != df['last_price'].shift()
        flatline_groups = price_changed.cumsum()
        streak_times = df.groupby(flatline_groups)['event_time'].agg(['min', 'max'])
        streak_durations = (streak_times['max'] - streak_times['min']).dt.total_seconds()

        stale_streaks = streak_durations[streak_durations >= STALE_THRESHOLD_SEC]
        if not stale_streaks.empty:
            longest_stale = stale_streaks.max()
            warnings.append(
                f"Stale data! 'last_price' flatlined {len(stale_streaks)} times "
                f"for >= 10m. Longest flatline: {longest_stale/60:.1f}m."
            )

            # --- NEW: Log the exact time windows for the flatlines ---
            stale_details = streak_times.loc[stale_streaks.index]
            for _, row in stale_details.iterrows():
                start_str = row['min'].strftime('%Y-%m-%d %H:%M:%S UTC')
                end_str = row['max'].strftime('%Y-%m-%d %H:%M:%S UTC')
                duration_mins = (row['max'] - row['min']).total_seconds() / 60

                warnings.append(
                    f"  -> Flatline incident: {start_str} to {end_str} ({duration_mins:.1f} mins)"
                )

    # ---------------------------------------------------------
    # 3. CLICKHOUSE SCHEMA VALIDATION (Pandas-Native Approach)
    # ---------------------------------------------------------
    try:
        logger.info("Connecting to ClickHouse to verify schema...")
        client = clickhouse_connect.get_client(
            host=ch_host, port=ch_port, username=ch_user, password=ch_pass
        )

        # Asking ClickHouse for a Pandas DataFrame directly strips away
        # all the weird tuple/list formatting issues.
        ch_schema_df = client.query_df(f"DESCRIBE TABLE {ch_table}")

        # Extract clean strings from the 'name' column
        ch_col_names = ch_schema_df['name'].tolist()
        pq_col_names = df.columns.tolist()

        # Check 1: Fatal Error - Parquet has extra columns
        missing_in_ch = set(pq_col_names) - set(ch_col_names)
        if missing_in_ch:
            errors.append(f"Parquet has columns that ClickHouse doesn't know about: {missing_in_ch}")

        # Check 2: Warning - ClickHouse has extra columns (assuming DEFAULTs)
        extra_in_ch = set(ch_col_names) - set(pq_col_names)
        if extra_in_ch:
            warnings.append(f"ClickHouse has {len(extra_in_ch)} extra columns. Assuming DEFAULT values will be used.")

    except Exception as e:
        errors.append(f"Failed to fetch ClickHouse schema for validation: {e}")

    # ---------------------------------------------------------
    # REPORTING
    # ---------------------------------------------------------
    for w in warnings:
        logger.warning(f"VALIDATION WARNING: {w}")

    if errors:
        logger.error("Data failed validation checks!")
        for e in errors:
            logger.error(f"VALIDATION ERROR: {e}")
        return False

    logger.info("All validation checks passed. Data is safe to load.")
    return True
