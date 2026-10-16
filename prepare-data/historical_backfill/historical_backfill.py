import pandas as pd
import requests
import zipfile
import io
import time
from datetime import datetime, timedelta, timezone
import clickhouse_connect

# --- CONFIGURATION ---
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
SCRAPE_INTERVAL_SECONDS = 15
SYMBOL = "LINKUSDC"
TARGET_TABLE = "binance_ww.raw_ticks_LINKUSDC_binance"
START_DATE = "2022-09-30"
END_DATE = "2023-03-23"

# Initialize ClickHouse Client
try:
    client = clickhouse_connect.get_client(
        host='10.1.19.249',
        port=8124,
        username='exporter',
        password='savqi9-mewces-nAcjod!-11',
        send_receive_timeout=120  # Add a 2-minute buffer
    )
    print("✅ Connected to ClickHouse")
except Exception as e:
    print(f"❌ Failed to connect to ClickHouse: {e}")
    exit(1)

def download_and_load_csv(date_str, symbol):
    """Downloads a single day of aggTrades and loads it into Pandas."""
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"

    print(f"  ⬇️ Downloading {filename}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"⚠️ Missing data for {date_str} (Status {response.status_code})")
            return pd.DataFrame()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, names=[
                    'agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match'
                ], usecols=['price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp'])
                return df
    except Exception as e:
        print(f"❌ Error downloading {date_str}: {e}")
        return pd.DataFrame()

def generate_daily_payload_vectorized(df_prev, df_curr, current_dt):
    """Replaces the slow loop with C-optimized Pandas vectorization."""
    df = pd.concat([df_prev, df_curr], ignore_index=True)
    if df.empty:
        return pd.DataFrame()

    print("  🧮 Running vectorized 24h math...", end="\r")

    df['quote_qty'] = df['price'] * df['quantity']

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    resampled = df.resample('15s').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('quantity', 'sum'),
        last_qty=('quantity', 'last'),
        quote_volume=('quote_qty', 'sum'),
        first_trade_id=('first_trade_id', 'first'),
        last_trade_id=('last_trade_id', 'last')
    )

    resampled['close'] = resampled['close'].ffill()
    resampled['open'] = resampled['open'].fillna(resampled['close'])
    resampled['high'] = resampled['high'].fillna(resampled['close'])
    resampled['low'] = resampled['low'].fillna(resampled['close'])
    resampled['volume'] = resampled['volume'].fillna(0)
    resampled['quote_volume'] = resampled['quote_volume'].fillna(0)
    resampled['last_qty'] = resampled['last_qty'].fillna(0)

    # FIX: Force any remaining NaNs in the earliest buckets to 0
    resampled['last_trade_id'] = resampled['last_trade_id'].ffill().fillna(0)
    resampled['first_trade_id'] = resampled['first_trade_id'].fillna(resampled['last_trade_id']).fillna(0)

    window = 5760
    rolling = pd.DataFrame(index=resampled.index)

    rolling['open_price'] = resampled['open'].shift(window - 1)
    rolling['high_price'] = resampled['high'].rolling(window).max()
    rolling['low_price'] = resampled['low'].rolling(window).min()
    rolling['last_price'] = resampled['close']
    rolling['volume'] = resampled['volume'].rolling(window).sum()
    rolling['quote_volume'] = resampled['quote_volume'].rolling(window).sum()
    rolling['last_qty'] = resampled['last_qty']
    rolling['first_trade_id'] = resampled['first_trade_id'].shift(window - 1)
    rolling['last_trade_id'] = resampled['last_trade_id']

    start_time = pd.to_datetime(current_dt)
    end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    target_day_df = rolling[start_time:end_time].copy()

    target_day_df.dropna(subset=['open_price'], inplace=True)

    if target_day_df.empty:
        return pd.DataFrame()

    target_day_df['price_change'] = target_day_df['last_price'] - target_day_df['open_price']

    target_day_df['price_change_percent'] = 0.0
    mask_open = target_day_df['open_price'] > 0
    target_day_df.loc[mask_open, 'price_change_percent'] = (target_day_df['price_change'] / target_day_df['open_price']) * 100.0

    target_day_df['weighted_avg_price'] = target_day_df['last_price']
    mask_vol = target_day_df['volume'] > 0
    target_day_df.loc[mask_vol, 'weighted_avg_price'] = target_day_df['quote_volume'] / target_day_df['volume']

    target_day_df['trade_count'] = target_day_df['last_trade_id'] - target_day_df['first_trade_id'] + 1

    # FIX: If no trades happened during a gap, ensure trade count is strictly 0
    target_day_df.loc[target_day_df['last_trade_id'] == 0, 'trade_count'] = 0

    target_day_df['exchange'] = "binance"
    target_day_df['symbol'] = SYMBOL
    target_day_df['bid_price'] = 0.0
    target_day_df['ask_price'] = 0.0
    target_day_df['bid_qty'] = 0.0
    target_day_df['ask_qty'] = 0.0
    target_day_df['prev_close_price'] = target_day_df['open_price']

    target_day_df['event_time'] = target_day_df.index
    target_day_df['close_time'] = target_day_df.index
    target_day_df['open_time'] = target_day_df.index - pd.Timedelta(days=1)

    float_cols = ['last_price', 'bid_price', 'ask_price', 'bid_qty', 'ask_qty', 'last_qty',
                    'open_price', 'high_price', 'low_price', 'prev_close_price', 'weighted_avg_price',
                    'price_change', 'price_change_percent', 'volume', 'quote_volume']
    for col in float_cols:
        target_day_df[col] = target_day_df[col].fillna(0.0).round(8).astype(float)

    int_cols = ['trade_count', 'first_trade_id', 'last_trade_id']
    for col in int_cols:
        # FIX: Fill missing values with 0, cast to int, and clip at 0 to guarantee UInt64 safety
        target_day_df[col] = target_day_df[col].fillna(0).astype(int).clip(lower=0)

    print("  ✅ 24h math complete.       ")
    return target_day_df
def insert_with_retry(df_batch, label, max_retries=3):
    """Inserts batch DataFrame directly into ClickHouse."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🚀 Pushing {len(df_batch)} rows for {label} to ClickHouse (Attempt {attempt})...")
            client.insert_df(TARGET_TABLE, df_batch)
            print(f"✅ Successfully inserted {label} data.")
            return True
        except Exception as e:
            print(f"⚠ Insert failed: {e}")
            if attempt < max_retries:
                sleep_time = attempt * 5
                print(f"⏳ Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"❌ FATAL: Failed to insert {label}.")
                return False

def process_historical_bulk(start_date_str, end_date_str):
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    prev_date = start_dt - timedelta(days=1)
    df_prev = download_and_load_csv(prev_date.strftime("%Y-%m-%d"), SYMBOL)

    current_dt = start_dt

    print(f"\n🚀 Starting Vectorized Bulk Processing from {start_date_str} to {end_date_str}")

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        df_curr = download_and_load_csv(date_str, SYMBOL)

        if df_prev.empty or df_curr.empty:
            print(f"⏭ Skipping {date_str} due to missing data.")
        else:
            daily_df = generate_daily_payload_vectorized(df_prev, df_curr, current_dt)
            if not daily_df.empty:
                # INSERT IMMEDIATELY INSTEAD OF BUFFERING
                insert_success = insert_with_retry(daily_df, date_str)
                if not insert_success:
                    print(f"🛑 Halting script due to insertion failure on {date_str}.")
                    break

        df_prev = df_curr
        current_dt += timedelta(days=1)

if __name__ == "__main__":
    process_historical_bulk(START_DATE, END_DATE)