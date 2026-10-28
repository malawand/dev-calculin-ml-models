import pandas as pd
import clickhouse_connect

# ---------------------------------------------------------
# 1. Fetch Price Data from ClickHouse
# ---------------------------------------------------------
print("Connecting to ClickHouse...")

# Replace with your actual ClickHouse credentials
client = clickhouse_connect.get_client(
    host='10.1.19.249', # or your database IP/URL
    username='exporter', 
    password='savqi9-mewces-nAcjod!-11',
    port=8124 
)

# Note: The end date was updated to 2026-08-08 to match your Cortex data
query = """
SELECT
    toStartOfInterval(event_time, INTERVAL 1 HOUR) AS time_bucket,
    argMax(last_price, event_time) AS close_price
FROM binance_ww.raw_ticks_binance
WHERE symbol = 'BTCUSDT' 
  AND event_time >= '2023-03-24'
  AND event_time <= '2026-08-03'
GROUP BY time_bucket
ORDER BY time_bucket
"""

print("Executing query...")
prices_df = client.query_df(query)
print(f"Retrieved {len(prices_df)} rows of price data.")

# ---------------------------------------------------------
# 2. Load Your Cortex Features Data
# ---------------------------------------------------------
print("Loading Cortex data...")
cortex_file_path = "btc_cortex_features.csv"
features_df = pd.read_csv(cortex_file_path)

# ---------------------------------------------------------
# 3. Standardize the Timezones
# ---------------------------------------------------------
print("Standardizing timezones for merge...")

# Convert both columns to pandas datetime objects and enforce UTC timezone
features_df["time_bucket"] = pd.to_datetime(features_df["time_bucket"], utc=True)
prices_df["time_bucket"] = pd.to_datetime(prices_df["time_bucket"], utc=True)

# ---------------------------------------------------------
# 4. Merge the Datasets
# ---------------------------------------------------------
print("Merging datasets...")

# A 'left' merge keeps all your feature rows and aligns the close_price where the hours match
combined_df = pd.merge(features_df, prices_df, on="time_bucket", how="left")

# Save the final combined dataset so you don't have to query ClickHouse every time
output_filename = "cortex_features_with_prices.csv"
combined_df.to_csv(output_filename, index=False)

print("\n" + "=" * 50)
print(f"Success! Saved combined data to {output_filename}")
print("=" * 50)
print("\nPreview of the merged data:")
print(combined_df.head())