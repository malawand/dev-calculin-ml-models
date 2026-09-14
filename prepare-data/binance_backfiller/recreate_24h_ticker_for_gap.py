import pandas as pd
import requests
import zipfile
import io
import json
import random
from datetime import datetime, timedelta, timezone

# Binance Vision URLs for daily aggTrades
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
SCRAPE_INTERVAL_SECONDS = 15  # UPDATED: Prometheus now scrapes every 15s

def download_and_load_csv(date_str, symbol):
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    
    print(f"  ⬇️ Downloading {filename}...")
    response = requests.get(url)
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        print(f"⚠️ Warning: Could not download {filename} (Status {response.status_code})")
        return pd.DataFrame() # Return empty if missing

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            headers = [
                'agg_trade_id', 'price', 'quantity', 'first_trade_id', 
                'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match'
            ]
            print(f"  🧠 Loading {date_str} into Pandas...")
            df = pd.read_csv(f, names=headers, usecols=[
                'agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp'
            ])

            # --- THE 2025+ BINANCE FIX ---
            # Starting Jan 1, 2025, Binance changed timestamps from milliseconds to microseconds.
            # If the timestamp is longer than 13 digits, we floor-divide by 1000 to normalize it back to ms!
            if not df.empty and df['timestamp'].iloc[0] > 10000000000000:
                df['timestamp'] = df['timestamp'] // 1000
                
            return df

def calculate_ticker_math(df_combined, close_time_dt, symbol):
    """Slices the 24h window based on close_time and runs the math."""
    end_ms = int(close_time_dt.timestamp() * 1000)
    start_ms = int((close_time_dt - timedelta(days=1)).timestamp() * 1000)
    
    # Slice the 24h window
    df_window = df_combined[(df_combined['timestamp'] >= start_ms) & (df_combined['timestamp'] <= end_ms)]
    
    if df_window.empty:
        return None

    return {
        "symbol": symbol,
        "open_time": (close_time_dt - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "close_time": close_time_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "last_price": df_window.iloc[-1]['price'],
        "last_qty": df_window.iloc[-1]['quantity'],
        "open_price": df_window.iloc[0]['price'],
        "high_price": df_window['price'].max(),
        "low_price": df_window['price'].min(),
        "volume": round(df_window['quantity'].sum(), 5),
        "quote_volume": round((df_window['price'] * df_window['quantity']).sum(), 2),
        "trade_count": int((df_window.iloc[-1]['last_trade_id'] - df_window.iloc[0]['first_trade_id']) + 1),
        "first_trade_id": int(df_window.iloc[0]['first_trade_id']),
        "last_trade_id": int(df_window.iloc[-1]['last_trade_id'])
    }

def run_qa_check(gap_rows, df_combined):
    """Randomly samples up to 3 generated rows and cross-checks them with the raw CSV data."""
    if not gap_rows:
        return
    
    samples = random.sample(gap_rows, min(3, len(gap_rows)))
    print("\n  🕵️  Automated QA Check (Random Samples):")
    
    for row in samples:
        close_time_str = row['close_time']
        # Convert string back to UTC datetime (adding "000" to ensure 6-digit microsecond parsing)
        dt = datetime.strptime(close_time_str + "000", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        target_ms = int(dt.timestamp() * 1000)
        
        # Look up the last trade that occurred exactly at or before this millisecond in the CSV
        df_filtered = df_combined[df_combined['timestamp'] <= target_ms]
        
        if not df_filtered.empty:
            csv_price = df_filtered.iloc[-1]['price']
            csv_trade_id = int(df_filtered.iloc[-1]['agg_trade_id'])
            csv_exact_ms = int(df_filtered.iloc[-1]['timestamp'])
            
            json_price = row['last_price']
            
            match_icon = "✅" if csv_price == json_price else "❌"
            print(f"    - Target Time: {close_time_str} | Generated Price: {json_price} | {match_icon}")
            print(f"      ↳ DETAILS: Found CSV Trade ID {csv_trade_id} exactly at {csv_exact_ms} ms (CSV Price: {csv_price})")
        else:
            print(f"    - Target Time: {close_time_str} | ⚠️ No CSV data found to compare.")
    print(" ") # Blank line for readability

def process_gaps(json_filepath):
    with open(json_filepath, 'r') as f:
        gaps = json.load(f)

    all_backfilled_rows = []

    for gap in gaps:
        symbol = gap['symbol']
        start_dt = datetime.fromisoformat(gap['start_str'])
        end_dt = datetime.fromisoformat(gap['end_str'])
        delay_seconds = gap.get('scrape_delay_seconds', 10) # Fallback to 10 if missing
        
        print(f"\n🚀 Processing Gap: {start_dt} to {end_dt} (Delay Offset: {delay_seconds}s)")

        # --- NEW LOGIC: Dynamic CSV downloading ---
        # To cover a 24h rolling window that crosses midnights, we need every calendar day 
        # from (start_dt - 1 day) up to the actual end_dt.
        earliest_date = (start_dt - timedelta(days=1)).date()
        latest_date = end_dt.date()
        
        required_dates = []
        curr_date = earliest_date
        while curr_date <= latest_date:
            required_dates.append(curr_date.strftime("%Y-%m-%d"))
            curr_date += timedelta(days=1)
            
        print(f"  📅 Downloading {len(required_dates)} days of data to cover midnight boundaries: {required_dates}")
        
        dfs = []
        for d_str in required_dates:
            df_day = download_and_load_csv(d_str, symbol)
            if not df_day.empty:
                dfs.append(df_day)

        if not dfs:
            print(f"❌ Skipping gap, no CSV data available.")
            continue
            
        df_combined = pd.concat(dfs, ignore_index=True)

        # --- EXCLUDE START AND END BOUNDARIES ---
        # Step forward 1 interval so we don't overwrite the healthy gap_start
        current_event_dt = start_dt + timedelta(seconds=SCRAPE_INTERVAL_SECONDS)
        gap_specific_rows = [] # Store rows just for this gap for the QA check

        # Loop through the gap, stopping strictly BEFORE the healthy gap_end
        while current_event_dt < end_dt:
            # 1. Determine the true market close_time using the offset
            true_close_dt = current_event_dt - timedelta(seconds=delay_seconds)

            # 2. Run heavy Pandas math for every row (since there are no more frozen blocks)
            print(f"  🧮 Calculating 24h math for close_time: {true_close_dt}", end="\r")
            calculated_payload = calculate_ticker_math(df_combined, true_close_dt, symbol)

            # 3. Create the row
            if calculated_payload:
                row = calculated_payload.copy()
                row['event_time'] = current_event_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                all_backfilled_rows.append(row)
                gap_specific_rows.append(row)
            
            # Step forward 15 seconds
            current_event_dt += timedelta(seconds=SCRAPE_INTERVAL_SECONDS)

        # Run automated QA for the gap we just finished!
        run_qa_check(gap_specific_rows, df_combined)

    # Save all backfilled data to JSON (ready for ClickHouse insertion)
    with open("ready_for_clickhouse.json", "w") as f:
        json.dump(all_backfilled_rows, f, indent=4)
        
    print(f"\n✅ Finished! Generated {len(all_backfilled_rows)} rows. Saved to 'ready_for_clickhouse.json'")

if __name__ == "__main__":
    # Ensure you have a pending_gaps.json file from script #1 before running!
    process_gaps("pending_gaps.json")
