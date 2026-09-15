import clickhouse_connect
import subprocess
from datetime import datetime, timezone, timedelta
import pandas as pd
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
CH_HOST = '10.1.19.249'
CH_PORT = 8124
CH_USER = 'exporter'
CH_PASS = 'savqi9-mewces-nAcjod!-11'
CH_DB = 'binance_ww'

SYMBOL = 'BTCUSDT'
EXTRACTION_BATCH_ID = f"ch_ols_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
HORIZONS = {
    '1h': 240, 
    '2h': 480, 
    '4h': 960, 
    '8h': 1920, 
    '16h': 3840, 
    '24h': 5760, 
    '48h': 11520, 
    '7d': 40320
}

def get_git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except Exception:
        return "deriv_ch_ols_log_prod_v2"

FEATURE_CODE_SHA = get_git_sha()

# ==========================================
# 2. CLICKHOUSE CONNECTION & DYNAMIC DATES
# ==========================================
print(f"[{datetime.now(timezone.utc)}] Connecting to ClickHouse...")
client = clickhouse_connect.get_client(
    host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASS,
    connect_timeout=120, send_receive_timeout=1800, compress='zstd'
)

print("Fetching dynamic date boundaries...")
last_processed_time = client.query(f"SELECT max(event_time) FROM {CH_DB}.features WHERE symbol = '{SYMBOL}'").result_rows[0][0]
latest_raw_time = client.query(f"SELECT max(event_time) FROM {CH_DB}.raw_ticks_binance WHERE symbol = '{SYMBOL}'").result_rows[0][0]

if last_processed_time.year == 1970:
    print("Features table is empty. Running full historical backfill...")
    first_raw_time = client.query(f"SELECT min(event_time) FROM {CH_DB}.raw_ticks_binance WHERE symbol = '{SYMBOL}'").result_rows[0][0]
    START_DATE = first_raw_time.strftime('%Y-%m-%d %H:%M:%S')
    loop_start_dt = pd.to_datetime(START_DATE)
else:
    print("Existing features found. Running incremental update...")
    loop_start_dt = pd.to_datetime(last_processed_time)

overall_end_dt = pd.to_datetime(latest_raw_time)

client.command(f"""
CREATE TABLE IF NOT EXISTS {CH_DB}.features_staging (
    symbol LowCardinality(String),
    event_time DateTime64(3, 'UTC'),
    ingestion_time DateTime64(3, 'UTC') DEFAULT now64(3),
    feature_name LowCardinality(String),
    value Float64,
    feature_code_sha LowCardinality(String),
    extraction_batch_id String
) ENGINE = MergeTree() ORDER BY (symbol, event_time, feature_name);
""")

client.command(f"""
CREATE TABLE IF NOT EXISTS {CH_DB}.features (
    symbol LowCardinality(String),
    event_time DateTime64(3, 'UTC'),
    ingestion_time DateTime64(3, 'UTC') DEFAULT now64(3),
    feature_name LowCardinality(String),
    value Float64,
    feature_code_sha LowCardinality(String)
) ENGINE = ReplacingMergeTree(ingestion_time) PARTITION BY toYYYYMM(event_time) 
ORDER BY (symbol, event_time, feature_code_sha, feature_name);
""")

# ==========================================
# 3. COMPUTE OLS & IN-DATABASE UNPIVOT
# ==========================================
print("Generating ClickHouse Native OLS and Unpivot statements...")
ch_ols_1st_selects = []
ch_ols_2nd_selects = []
first_deriv_names = []
unpivot_pairs = []

for horizon_name, window in HORIZONS.items():
    col_1st = f"deriv_{horizon_name}"
    col_2nd = f"deriv_prime_{horizon_name}"
    first_deriv_names.append(col_1st)
    
    # Build array elements for native unpivoting
    unpivot_pairs.append(f"('{col_1st}', {col_1st})")
    unpivot_pairs.append(f"('{col_2nd}', {col_2nd})")
    
    if window >= 2:
        ch_ols_1st_selects.append(f"""
        tupleElement(
            simpleLinearRegression(toUnixTimestamp(time_bucket) , raw_price) 
            OVER (ORDER BY time_bucket ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW), 
        1) AS {col_1st}
        """)
        
        ch_ols_2nd_selects.append(f"""
        tupleElement(
            simpleLinearRegression(toUnixTimestamp(time_bucket) , {col_1st}) 
            OVER (ORDER BY time_bucket ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW), 
        1) AS {col_2nd}
        """)
    else:
        ch_ols_1st_selects.append(f"0.0 AS {col_1st}")
        ch_ols_2nd_selects.append(f"0.0 AS {col_2nd}")

array_join_str = "arrayJoin([" + ", ".join(unpivot_pairs) + "]) AS feature_tuple"

CHUNK_DAYS = 7
current_start = loop_start_dt

print(f"Beginning direct-to-database pipeline extraction from {current_start} to {overall_end_dt}...")

while current_start < overall_end_dt:
    current_end = min(current_start + pd.Timedelta(days=CHUNK_DAYS), overall_end_dt)
    window_start = current_start - pd.Timedelta(days=7)
    
    print(f"Executing and natively unpivoting: {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}...")
    
    insert_query = f"""
    INSERT INTO {CH_DB}.features_staging (symbol, event_time, ingestion_time, feature_name, value, feature_code_sha, extraction_batch_id)
    WITH raw_bars AS (
        SELECT
            toStartOfInterval(event_time, INTERVAL 15 SECOND) AS time_bucket,
            argMax(last_price, event_time) AS raw_price
        FROM {CH_DB}.raw_ticks_binance
        WHERE symbol = '{SYMBOL}' 
          AND event_time >= '{window_start.strftime('%Y-%m-%d %H:%M:%S')}' 
          AND event_time <= '{current_end.strftime('%Y-%m-%d %H:%M:%S')}'
        GROUP BY time_bucket
    ),
    first_derivative AS (
        SELECT 
            time_bucket, 
            raw_price,
            {",".join(ch_ols_1st_selects)}
        FROM raw_bars 
    ),
    wide_features AS (
        SELECT 
            time_bucket, 
            {", ".join(first_deriv_names)},
            {",".join(ch_ols_2nd_selects)}
        FROM first_derivative 
        WHERE time_bucket > '{current_start.strftime('%Y-%m-%d %H:%M:%S')}'
          AND time_bucket <= '{current_end.strftime('%Y-%m-%d %H:%M:%S')}'
    )
    SELECT
        '{SYMBOL}' AS symbol,
        time_bucket AS event_time,
        now64(3) AS ingestion_time,
        feature_tuple.1 AS feature_name,
        toFloat64(feature_tuple.2) AS value,
        '{FEATURE_CODE_SHA}' AS feature_code_sha,
        '{EXTRACTION_BATCH_ID}' AS extraction_batch_id
    FROM (
        SELECT 
            time_bucket,
            {array_join_str}
        FROM wide_features
    )
    """
    
    # Executes the entire pipeline logic server-side
    client.command(insert_query)
    current_start = current_end

# ==========================================
# 4. VERIFICATION & PROMOTION
# ==========================================
print("\nPipeline processing finished. Verifying staging table integrity...")

staging_count = client.command(f"""
    SELECT count() 
    FROM {CH_DB}.features_staging 
    WHERE extraction_batch_id = '{EXTRACTION_BATCH_ID}'
""")

if staging_count == 0:
    print("No new data was inserted. Aborting pipeline gracefully.")
    sys.exit(0)

print(f"Verified {staging_count:,} total feature rows. Promoting payload to main features table...")
client.command(f"""
INSERT INTO {CH_DB}.features (symbol, event_time, ingestion_time, feature_name, value, feature_code_sha)
SELECT symbol, event_time, ingestion_time, feature_name, value, feature_code_sha
FROM {CH_DB}.features_staging 
WHERE extraction_batch_id = '{EXTRACTION_BATCH_ID}'
""")

client.command(f"ALTER TABLE {CH_DB}.features_staging DELETE WHERE extraction_batch_id = '{EXTRACTION_BATCH_ID}'")
print("Extraction and ingestion complete.")