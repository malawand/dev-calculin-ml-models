#!/usr/bin/env python3
"""
Fill Data Gaps with Yahoo Finance

Check for gaps in Prometheus data and fill with Yahoo Finance prices
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path


def detect_gaps(df, max_gap_minutes=30):
    """
    Detect gaps in time series data
    
    Args:
        df: DataFrame with DatetimeIndex
        max_gap_minutes: Maximum acceptable gap
        
    Returns:
        List of (start_time, end_time) tuples for gaps
    """
    gaps = []
    df = df.sort_index()
    
    for i in range(len(df) - 1):
        current_time = df.index[i]
        next_time = df.index[i + 1]
        gap = (next_time - current_time).total_seconds() / 60
        
        if gap > max_gap_minutes:
            gaps.append((current_time, next_time, gap))
    
    return gaps


def fill_gaps_with_yahoo(df, gaps, symbol='BTC-USD'):
    """
    Fill gaps using Yahoo Finance data
    
    Args:
        df: DataFrame with gaps
        gaps: List of (start, end, gap_size) tuples
        symbol: Yahoo Finance symbol
        
    Returns:
        DataFrame with gaps filled
    """
    if not gaps:
        return df
    
    print(f"🔍 Found {len(gaps)} gaps in data")
    
    df_filled = df.copy()
    
    for i, (start, end, gap_minutes) in enumerate(gaps, 1):
        print(f"   Gap {i}/{len(gaps)}: {start} → {end} ({gap_minutes:.0f} minutes)")
        
        # Fetch Yahoo Finance data for gap period
        try:
            # Add buffer to ensure we get data
            fetch_start = start - timedelta(hours=1)
            fetch_end = end + timedelta(hours=1)
            
            yf_data = yf.download(
                symbol,
                start=fetch_start,
                end=fetch_end,
                interval='15m',
                progress=False
            )
            
            if yf_data.empty:
                print(f"      ⚠️  No Yahoo Finance data available")
                continue
            
            # Get prices in the gap
            gap_mask = (yf_data.index > start) & (yf_data.index < end)
            gap_data = yf_data[gap_mask]
            
            if gap_data.empty:
                print(f"      ⚠️  No data in gap period")
                continue
            
            # For each timestamp in gap, add price
            for timestamp in gap_data.index:
                if timestamp not in df_filled.index:
                    # Create row with Yahoo Finance close price
                    new_row = pd.Series(index=df_filled.columns, dtype=float)
                    new_row['crypto_last_price'] = gap_data.loc[timestamp, 'Close']
                    
                    # Fill other columns with interpolation markers
                    for col in df_filled.columns:
                        if col != 'crypto_last_price':
                            new_row[col] = np.nan
                    
                    df_filled.loc[timestamp] = new_row
            
            print(f"      ✅ Filled {len(gap_data)} timestamps")
            
        except Exception as e:
            print(f"      ❌ Error filling gap: {e}")
    
    # Sort and interpolate other columns
    df_filled = df_filled.sort_index()
    
    # Forward fill then backward fill for non-price columns
    for col in df_filled.columns:
        if col != 'crypto_last_price':
            df_filled[col] = df_filled[col].ffill().bfill()
    
    return df_filled


def main():
    """Check and fill data gaps"""
    
    print("="*80)
    print("🔍 CHECKING FOR DATA GAPS")
    print("="*80)
    
    # Load combined dataset
    data_path = Path("../btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet")
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return
    
    print(f"\n📥 Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    print(f"   Loaded {len(df)} samples")
    print(f"   Date range: {df.index[0]} → {df.index[-1]}")
    print(f"   Columns: {len(df.columns)}")
    
    # Detect gaps
    print(f"\n🔍 Detecting gaps (>30 minutes)...")
    gaps = detect_gaps(df, max_gap_minutes=30)
    
    if not gaps:
        print("   ✅ No significant gaps found!")
        return
    
    # Fill gaps
    print(f"\n📝 Filling gaps with Yahoo Finance data...")
    df_filled = fill_gaps_with_yahoo(df, gaps)
    
    # Save filled dataset
    output_path = Path("../btc_direction_predictor/artifacts/historical_data/combined_full_dataset_filled.parquet")
    df_filled.to_parquet(output_path)
    
    print(f"\n✅ Filled dataset saved: {output_path}")
    print(f"   Original samples: {len(df)}")
    print(f"   Filled samples: {len(df_filled)}")
    print(f"   Added: {len(df_filled) - len(df)} samples")
    
    print("\n" + "="*80)
    print("✅ Gap filling complete!")
    print("="*80)


if __name__ == "__main__":
    main()



