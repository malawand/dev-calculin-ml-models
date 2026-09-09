import clickhouse_connect
import pandas as pd
import subprocess
from datetime import datetime, timezone
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
START_DATE = '2023-03-24 00:00:00'
END_DATE = '2026-08-01 00:00:00'
EXTRACTION_BATCH_ID = f"ch_ols_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
HORIZONS = {'1h': 4, '2h': 8, '4h': 16, '8h': 32, '16h' : 64 , '24h': 96, '48h': 192, '7d': 672}

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
# 2. CLICKHOUSE CONNECTION
# ==========================================
print(f"[{datetime.now(timezone.utc)}] Connecting to ClickHouse...")
client = clickhouse_connect.get_client(
    host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASS,
    connect_timeout=60, send_receive_timeout=300, compress='zstd'
)

# Ensure tables exist
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
# 3. COMPUTE CLICKHOUSE OLS (1st & 2nd Deriv)
# ==========================================
print("Computing 1st and 2nd derivatives inside ClickHouse...")
ch_ols_1st_selects = []
ch_ols_2nd_selects = []
first_deriv_names = []

for horizon_name, window in HORIZONS.items():
    col_1st = f"deriv_{horizon_name}"
    col_2nd = f"deriv_prime_{horizon_name}"
    first_deriv_names.append(col_1st)
    
    if window >= 2:
        # Phase 1: Velocity
        ch_ols_1st_selects.append(f"""
        tupleElement(
            simpleLinearRegression(toUnixTimestamp(time_bucket) / 900, log_price) 
            OVER (ORDER BY time_bucket ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW), 
        1) AS {col_1st}
        """)
        
        # Phase 2: Acceleration (Uses the Phase 1 output as input)
        ch_ols_2nd_selects.append(f"""
        tupleElement(
            simpleLinearRegression(toUnixTimestamp(time_bucket) / 900, {col_1st}) 
            OVER (ORDER BY time_bucket ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW), 
        1) AS {col_2nd}
        """)
    else:
        ch_ols_1st_selects.append(f"0.0 AS {col_1st}")
        ch_ols_2nd_selects.append(f"0.0 AS {col_2nd}")

# Construct the CTE stacked query
query_raw = f"""
WITH raw_bars AS (
    SELECT
        toStartOfInterval(event_time, INTERVAL 15 MINUTE) AS time_bucket,
        log(argMax(last_price, event_time)) AS log_price
    FROM {CH_DB}.raw_ticks_binance
    WHERE symbol = '{SYMBOL}' 
      AND event_time >= '{START_DATE}' 
      AND event_time < '{END_DATE}'
    GROUP BY time_bucket
),
first_derivative AS (
    SELECT 
        time_bucket, 
        log_price,
        {",".join(ch_ols_1st_selects)}
    FROM raw_bars 
)
SELECT 
    time_bucket, 
    {", ".join(first_deriv_names)},
    {",".join(ch_ols_2nd_selects)}
FROM first_derivative 
ORDER BY time_bucket
"""

df_features = client.query_df(query_raw)
df_features = df_features.sort_values('time_bucket').reset_index(drop=True)

if df_features.empty:
    print("Error: No data retrieved. Aborting pipeline.")
    sys.exit(1)

# ==========================================
# 4. UNPIVOT & INSERT (CHUNKED)
# ==========================================
# This automatically grabs both deriv_* and deriv_prime_*
feature_cols = [col for col in df_features.columns if col.startswith('deriv_')]
df_long = df_features.melt(
    id_vars=['time_bucket'], 
    value_vars=feature_cols, 
    var_name='feature_name', 
    value_name='value'
)

df_staging = pd.DataFrame({
    'symbol': SYMBOL,
    'event_time': pd.to_datetime(df_long['time_bucket'], utc=True),
    'ingestion_time': pd.Timestamp.now(tz='UTC'),
    'feature_name': df_long['feature_name'],
    'value': df_long['value'],
    'feature_code_sha': FEATURE_CODE_SHA,
    'extraction_batch_id': EXTRACTION_BATCH_ID
}).dropna(subset=['value']).reset_index(drop=True)

total_rows = len(df_staging)
print(f"Prepared {total_rows:,} rows. Inserting into staging...")

CHUNK_SIZE = 100_000
total_inserted = 0

for start_idx in range(0, total_rows, CHUNK_SIZE):
    chunk = df_staging.iloc[start_idx:start_idx + CHUNK_SIZE]
    client.insert_df(f'{CH_DB}.features_staging', chunk)
    
    total_inserted += len(chunk)
    print(f"  -> Inserted {total_inserted:,} / {total_rows:,} rows...")

print("Verifying staging table integrity...")
staging_count = client.command(f"""
    SELECT count() 
    FROM {CH_DB}.features_staging 
    WHERE extraction_batch_id = '{EXTRACTION_BATCH_ID}'
""")

if staging_count != total_rows:
    print(f"Validation FAILED: Staging has {staging_count} rows, expected {total_rows}.")
    client.command(f"ALTER TABLE {CH_DB}.features_staging DELETE WHERE extraction_batch_id = '{EXTRACTION_BATCH_ID}'")
    sys.exit(1)

print("Promoting payload to main features table...")
client.command(f"""
INSERT INTO {CH_DB}.features (symbol, event_time, ingestion_time, feature_name, value, feature_code_sha)
SELECT symbol, event_time, ingestion_time, feature_name, value, feature_code_sha
FROM {CH_DB}.features_staging 
WHERE extraction_batch_id = '{EXTRACTION_BATCH_ID}'
""")

client.command(f"ALTER TABLE {CH_DB}.features_staging DELETE WHERE extraction_batch_id = '{EXTRACTION_BATCH_ID}'")
print("Extraction and ingestion complete.")