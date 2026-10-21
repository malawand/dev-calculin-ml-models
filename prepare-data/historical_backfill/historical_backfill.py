import argparse
import io
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
import clickhouse_connect
import pandas as pd
import requests
import re

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Historical Binance Vision 24h Ticker Multi-Symbol Pipeline into ClickHouse."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated symbols (e.g., 'BTCUSDT,ETHUSDC,LINKBTC') or single symbol.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="auto",
        help="Start date in YYYY-MM-DD format, or 'auto' to auto-detect the listing date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--scrape-interval",
        type=int,
        default=15,
        help="Snapshot interval in seconds (default: 15).",
    )
    parser.add_argument(
        "--ch-host",
        type=str,
        default="10.1.19.249",
        help="ClickHouse host (default: 10.1.19.249).",
    )
    parser.add_argument(
        "--ch-port",
        type=int,
        default=8124,
        help="ClickHouse port (default: 8124).",
    )
    parser.add_argument(
        "--ch-user",
        type=str,
        default="exporter",
        help="ClickHouse username (default: exporter).",
    )
    parser.add_argument(
        "--ch-password",
        type=str,
        required=True,
        help="ClickHouse password.",
    )

    args = parser.parse_args()
    # Normalize symbols: split by comma, strip whitespace, remove empty items
    args.symbol_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    return args

import re

def get_first_available_date(symbol):
    """Auto-detects the exact date Binance started archiving a symbol via S3."""
    print(f"  🔍 Auto-detecting earliest available date for {symbol}...")
    
    # Binance Vision's underlying S3 bucket URL
    url = f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=data/spot/daily/aggTrades/{symbol}/&max-keys=5"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Search the XML text for the first matching filename pattern
            match = re.search(rf"{symbol}-aggTrades-(\d{{4}}-\d{{2}}-\d{{2}})\.zip", response.text)
            if match:
                first_date = match.group(1)
                print(f"  🎯 Found earliest date: {first_date}")
                return first_date
            else:
                print(f"  ⚠️ No zip files found for {symbol} in the archive.")
        else:
            print(f"  ⚠️ S3 returned status {response.status_code}")
    except Exception as e:
        print(f"  ⚠️ Could not auto-detect date: {e}")
    
    return None

def download_and_load_csv(date_str, symbol):
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"

    print(f"  ⬇️ Downloading {filename}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"  ⚠️ Missing data for {date_str} (Status {response.status_code})")
            return pd.DataFrame()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(
                    f,
                    names=[
                        "agg_trade_id",
                        "price",
                        "quantity",
                        "first_trade_id",
                        "last_trade_id",
                        "timestamp",
                        "is_buyer_maker",
                        "is_best_match",
                    ],
                    usecols=[
                        "price",
                        "quantity",
                        "first_trade_id",
                        "last_trade_id",
                        "timestamp",
                    ],
                )
                return df
    except Exception as e:
        print(f"  ❌ Error downloading {date_str}: {e}")
        return pd.DataFrame()


def generate_daily_payload_vectorized(
    df_prev, df_curr, current_dt, symbol, interval_sec
):
    df = pd.concat([df_prev, df_curr], ignore_index=True)
    if df.empty:
        return pd.DataFrame()

    print("  🧮 Running vectorized 24h math...", end="\r")

    df["quote_qty"] = df["price"] * df["quantity"]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    bucket_freq = f"{interval_sec}s"
    resampled = df.resample(bucket_freq).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
        last_qty=("quantity", "last"),
        quote_volume=("quote_qty", "sum"),
        first_trade_id=("first_trade_id", "first"),
        last_trade_id=("last_trade_id", "last"),
    )

    resampled["close"] = resampled["close"].ffill()
    resampled["open"] = resampled["open"].fillna(resampled["close"])
    resampled["high"] = resampled["high"].fillna(resampled["close"])
    resampled["low"] = resampled["low"].fillna(resampled["close"])
    resampled["volume"] = resampled["volume"].fillna(0)
    resampled["quote_volume"] = resampled["quote_volume"].fillna(0)
    resampled["last_qty"] = resampled["last_qty"].fillna(0)

    resampled["last_trade_id"] = resampled["last_trade_id"].ffill().fillna(0)
    resampled["first_trade_id"] = (
        resampled["first_trade_id"]
        .fillna(resampled["last_trade_id"])
        .fillna(0)
    )

    window = int(86400 / interval_sec)
    rolling = pd.DataFrame(index=resampled.index)

    rolling["open_price"] = resampled["open"].shift(window - 1)
    rolling["high_price"] = resampled["high"].rolling(window).max()
    rolling["low_price"] = resampled["low"].rolling(window).min()
    rolling["last_price"] = resampled["close"]
    rolling["volume"] = resampled["volume"].rolling(window).sum()
    rolling["quote_volume"] = resampled["quote_volume"].rolling(window).sum()
    rolling["last_qty"] = resampled["last_qty"]
    rolling["first_trade_id"] = resampled["first_trade_id"].shift(window - 1)
    rolling["last_trade_id"] = resampled["last_trade_id"]

    start_time = pd.to_datetime(current_dt)
    end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    target_day_df = rolling[start_time:end_time].copy()

    target_day_df.dropna(subset=["open_price"], inplace=True)
    if target_day_df.empty:
        return pd.DataFrame()

    target_day_df["price_change"] = (
        target_day_df["last_price"] - target_day_df["open_price"]
    )

    target_day_df["price_change_percent"] = 0.0
    mask_open = target_day_df["open_price"] > 0
    target_day_df.loc[mask_open, "price_change_percent"] = (
        target_day_df["price_change"] / target_day_df["open_price"]
    ) * 100.0

    target_day_df["weighted_avg_price"] = target_day_df["last_price"]
    mask_vol = target_day_df["volume"] > 0
    target_day_df.loc[mask_vol, "weighted_avg_price"] = (
        target_day_df["quote_volume"] / target_day_df["volume"]
    )

    target_day_df["trade_count"] = (
        target_day_df["last_trade_id"] - target_day_df["first_trade_id"] + 1
    )
    target_day_df.loc[target_day_df["last_trade_id"] == 0, "trade_count"] = 0

    target_day_df["exchange"] = "binance"
    target_day_df["symbol"] = symbol
    target_day_df["bid_price"] = 0.0
    target_day_df["ask_price"] = 0.0
    target_day_df["bid_qty"] = 0.0
    target_day_df["ask_qty"] = 0.0
    target_day_df["prev_close_price"] = target_day_df["open_price"]

    target_day_df["event_time"] = target_day_df.index
    target_day_df["close_time"] = target_day_df.index
    target_day_df["open_time"] = target_day_df.index - pd.Timedelta(days=1)

    float_cols = [
        "last_price",
        "bid_price",
        "ask_price",
        "bid_qty",
        "ask_qty",
        "last_qty",
        "open_price",
        "high_price",
        "low_price",
        "prev_close_price",
        "weighted_avg_price",
        "price_change",
        "price_change_percent",
        "volume",
        "quote_volume",
    ]
    for col in float_cols:
        target_day_df[col] = (
            target_day_df[col].fillna(0.0).round(8).astype(float)
        )

    int_cols = ["trade_count", "first_trade_id", "last_trade_id"]
    for col in int_cols:
        target_day_df[col] = (
            target_day_df[col].fillna(0).astype(int).clip(lower=0)
        )

    print("  ✅ 24h math complete.        ")
    return target_day_df


def insert_with_retry(client, target_table, df_batch, label, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"  🚀 Pushing {len(df_batch)} rows for {label} to {target_table} (Attempt {attempt})..."
            )
            client.insert_df(target_table, df_batch)
            print(f"  ✅ Successfully inserted {label} data.")
            return True
        except Exception as e:
            print(f"  ⚠ Insert failed: {e}")
            if attempt < max_retries:
                sleep_time = attempt * 5
                print(f"  ⏳ Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"  ❌ FATAL: Failed to insert {label}.")
                return False 


def process_symbol_bulk(client, symbol, start_dt, end_dt, interval_sec):
    target_table = f"binance_ww.raw_ticks_{symbol}_binance"

    print(f"\n{'='*60}")
    print(f"▶️ Starting Backfill for Symbol: {symbol}")
    print(f"   Destination Table: {target_table}")
    print(f"{'='*60}")

    prev_date = start_dt - timedelta(days=1)
    df_prev = download_and_load_csv(prev_date.strftime("%Y-%m-%d"), symbol)

    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        df_curr = download_and_load_csv(date_str, symbol)

        if df_prev.empty or df_curr.empty:
            print(f"  ⏭ Skipping {date_str} due to missing data.")
        else:
            daily_df = generate_daily_payload_vectorized(
                df_prev, df_curr, current_dt, symbol, interval_sec
            )
            if not daily_df.empty:
                insert_success = insert_with_retry(
                    client, target_table, daily_df, date_str
                )
                if not insert_success:
                    print(f"🛑 FATAL: Halting pipeline on symbol {symbol} due to database error.")
                    return False

        df_prev = df_curr
        current_dt += timedelta(days=1)

    print(f"✅ Completed backfill for {symbol}\n")
    return True


def run_pipeline(args):
    try:
        client = clickhouse_connect.get_client(
            host=args.ch_host,
            port=args.ch_port,
            username=args.ch_user,
            password=args.ch_password,
            send_receive_timeout=120,
        )
        print(f"✅ Connected to ClickHouse ({args.ch_host}:{args.ch_port})")
    except Exception as e:
        print(f"❌ Failed to connect to ClickHouse: {e}")
        sys.exit(1)

def run_pipeline(args):
    try:
        client = clickhouse_connect.get_client(
            host=args.ch_host,
            port=args.ch_port,
            username=args.ch_user,
            password=args.ch_password,
            send_receive_timeout=120,
        )
        print(f"✅ Connected to ClickHouse ({args.ch_host}:{args.ch_port})")
    except Exception as e:
        print(f"❌ Failed to connect to ClickHouse: {e}")
        sys.exit(1)

    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"\nTarget Symbols ({len(args.symbol_list)}): {', '.join(args.symbol_list)}")
    print(f"Requested Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")

    for sym in args.symbol_list:
        # 1. Resolve Start Date per symbol
        actual_start_date_str = args.start_date
        if actual_start_date_str.lower() == "auto":
            detected_date = get_first_available_date(sym)
            if not detected_date:
                print(f"💥 Skipping {sym} — could not determine a start date.")
                continue
            actual_start_date_str = detected_date

        try:
            start_dt = datetime.strptime(actual_start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"❌ Invalid date format '{actual_start_date_str}' for {sym}. Expected YYYY-MM-DD.")
            continue

        # 2. Safety check: ensure start date precedes end date
        if start_dt > end_dt:
            print(f"⏭ Skipping {sym} — Earliest data ({actual_start_date_str}) is after requested end date ({args.end_date}).")
            continue

        # 3. Execute pipeline
        success = process_symbol_bulk(client, sym, start_dt, end_dt, args.scrape_interval)
        if not success:
            print(f"💥 Pipeline aborted on symbol {sym}.")
            sys.exit(1)

    print("🎉 All symbols processed successfully.")


if __name__ == "__main__":
    cli_args = parse_arguments()
    run_pipeline(cli_args)