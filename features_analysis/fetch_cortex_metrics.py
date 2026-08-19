import requests
import pandas as pd
import time
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURATION
# ==========================================
CORTEX_URL = "http://10.1.20.60:9009/prometheus/api/v1/query_range" 
HEADERS = {}
OUTPUT_FILE = "btc_cortex_features.csv"

QUERIES = {
    # Wrapped in max() to aggregate across all instances/pods
    '7d_derivative': 'max(derivative_transitions_2{interval="7d",order="99"})',
    '7d_derivative_prime': 'max(derivative_prime_transitions_2{interval="7d_prime24h",order="99"})',
    
    # NEW FEATURE: The macro weighted polynomial transition
    'poly_transition(_weighted24h48h7d)': 'max(polynomial_transition{interval="_weighted24h48h7d",order="90"})'
}

# The full 3.5 year data range
START_TIME_STR = "2023-03-24T00:00:00Z"
END_TIME_STR = "2026-08-03T00:00:00Z"

# Convert to UNIX timestamps for the API
START_TIME = int(pd.to_datetime(START_TIME_STR).timestamp())
END_TIME = int(pd.to_datetime(END_TIME_STR).timestamp())
STEP = "1h"
CHUNK_DAYS = 180 

# ==========================================
# 2. HELPER FUNCTION
# ==========================================
def fetch_metric_in_chunks(query_name, query_str, start_ts, end_ts, step=STEP, chunk_days=CHUNK_DAYS):
    all_dfs = []
    current_start = start_ts
    chunk_sec = chunk_days * 24 * 3600
    
    print(f"\nFetching '{query_name}'...")
    print(f"  Query: {query_str}")
    
    while current_start < end_ts:
        current_end = min(current_start + chunk_sec, end_ts)
        start_str = datetime.fromtimestamp(current_start, tz=timezone.utc).strftime('%Y-%m-%d')
        end_str = datetime.fromtimestamp(current_end, tz=timezone.utc).strftime('%Y-%m-%d')
        print(f"  --> Chunk: {start_str} to {end_str}...", end="", flush=True)
        
        params = {
            "query": query_str,
            "start": current_start,
            "end": current_end,
            "step": step
        }
        
        try:
            response = requests.get(CORTEX_URL, params=params, headers=HEADERS, timeout=30)
            if response.status_code != 200:
                print(" FAILED")
                raise Exception(f"Cortex Error [{response.status_code}]: {response.text}")
                
            data = response.json()
            results = data.get('data', {}).get('result', [])
            
            if results:
                values = results[0]['values']
                df = pd.DataFrame(values, columns=['timestamp', query_name])
                all_dfs.append(df)
                print(f" Done ({len(values)} points)")
            else:
                print(" No data returned")
                
        except requests.exceptions.RequestException as e:
            print(" FAILED")
            raise Exception(f"Connection error: {e}")
            
        current_start = current_end + 3600
        
    if not all_dfs:
        print(f"Warning: No data collected for '{query_name}'!")
        return pd.DataFrame(columns=['timestamp', query_name])
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['timestamp'])
    final_df['timestamp'] = final_df['timestamp'].astype(int)
    final_df[query_name] = final_df[query_name].astype(float)
    return final_df

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    print("==========================================")
    print(f" Fetching Full Range: 2023 to 2026 ")
    print("==========================================")
    
    dfs = []
    for col_name, query_str in QUERIES.items():
        df_metric = fetch_metric_in_chunks(col_name, query_str, START_TIME, END_TIME)
        if not df_metric.empty:
            dfs.append(df_metric)
            
    if not dfs:
        print("\nError: Failed to fetch any data.")
        return

    # Merge metrics
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='timestamp', how='outer')
        
    merged_df['time_bucket'] = pd.to_datetime(merged_df['timestamp'], unit='s', utc=True)
    merged_df = merged_df.set_index('time_bucket')
    
    print("\n--- Applying Perfect Hourly Grid ---")
    perfect_grid = pd.date_range(start=START_TIME_STR, end=END_TIME_STR, freq='h', tz='UTC')
    print(f"Targeting a continuous grid of {len(perfect_grid)} hourly rows.")
    
    merged_df = merged_df.reindex(perfect_grid)
    
# Forward-fill gaps (e.g., if a single hour is missed by the API)
    merged_df['7d_derivative'] = merged_df['7d_derivative'].ffill()
    merged_df['7d_derivative_prime'] = merged_df['7d_derivative_prime'].ffill()
    merged_df['poly_transition(_weighted24h48h7d)'] = merged_df['poly_transition(_weighted24h48h7d)'].ffill()
    
    # REMOVED .bfill()! 
    # Instead, we drop any rows that still have missing data (which will delete 2023-2025)
    merged_df = merged_df.dropna()
    
    merged_df = merged_df.reset_index().rename(columns={'index': 'time_bucket'})
    cols = ['time_bucket', '7d_derivative', '7d_derivative_prime', 'poly_transition(_weighted24h48h7d)']
    merged_df = merged_df[cols]
    
    merged_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n==========================================")
    print(" SUCCESS!")
    print(f"Saved {len(merged_df)} continuous rows to '{OUTPUT_FILE}'.")
    print("==========================================")

if __name__ == "__main__":
    main()