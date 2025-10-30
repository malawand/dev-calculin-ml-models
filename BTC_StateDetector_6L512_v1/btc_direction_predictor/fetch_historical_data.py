"""
Fetch historical data from Cortex in 3-month chunks and save to CSV.
This bypasses Cortex's query size limits by fetching in smaller batches.
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.prom import PrometheusClient
from src.data.build_dataset import DatasetBuilder


def fetch_data_in_chunks(
    start_date: str,
    end_date: str,
    chunk_months: int = 3,
    step: str = "15m",
    output_dir: str = "artifacts/historical_data"
):
    """
    Fetch data from Cortex in chunks and save to CSV files.
    
    Args:
        start_date: Start date in ISO format (e.g., "2024-01-01T00:00:00Z")
        end_date: End date in ISO format
        chunk_months: Size of each chunk in months (default 3)
        step: Data resolution (default "15m")
        output_dir: Directory to save CSV files
    """
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Parse dates
    start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate chunks
    chunks = []
    current = start
    while current < end:
        chunk_end = min(
            current + timedelta(days=chunk_months * 30),  # Approximate months
            end
        )
        chunks.append((current, chunk_end))
        current = chunk_end
    
    print(f"📊 Fetching data in {len(chunks)} chunks...")
    print(f"   Date range: {start_date} → {end_date}")
    print(f"   Chunk size: {chunk_months} months each")
    print(f"   Step: {step}")
    print()
    
    all_dataframes = []
    
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        chunk_name = f"chunk_{i}_{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}"
        csv_path = os.path.join(output_dir, f"{chunk_name}.csv")
        
        # Skip if already exists
        if os.path.exists(csv_path):
            print(f"✅ Chunk {i}/{len(chunks)}: {chunk_name} (cached)")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            all_dataframes.append(df)
            continue
        
        print(f"📥 Chunk {i}/{len(chunks)}: {chunk_name}")
        print(f"   Fetching: {chunk_start.isoformat()} → {chunk_end.isoformat()}")
        
        # Update config for this chunk
        # Remove timezone info and add Z suffix
        config['data']['start'] = chunk_start.replace(tzinfo=None).isoformat() + 'Z'
        config['data']['end'] = chunk_end.replace(tzinfo=None).isoformat() + 'Z'
        config['data']['step'] = step
        
        try:
            # Fetch data using DatasetBuilder
            # Create a temp parquet path
            temp_parquet = os.path.join(output_dir, f"{chunk_name}_temp.parquet")
            builder = DatasetBuilder(config=config)
            
            df = builder.build(output_path=temp_parquet)
            
            if df is not None and len(df) > 0:
                # Save chunk to CSV
                df.to_csv(csv_path)
                all_dataframes.append(df)
                print(f"   ✅ Saved {len(df)} samples to {csv_path}")
                
                # Clean up temp parquet
                if os.path.exists(temp_parquet):
                    os.remove(temp_parquet)
            else:
                print(f"   ⚠️  No data fetched for this chunk")
        
        except Exception as e:
            print(f"   ❌ Error fetching chunk: {e}")
            continue
        
        print()
    
    # Combine all chunks
    if all_dataframes:
        print(f"🔗 Combining {len(all_dataframes)} chunks...")
        combined_df = pd.concat(all_dataframes, axis=0)
        
        # Remove duplicates (in case of overlapping timestamps)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        combined_df.sort_index(inplace=True)
        
        # Save combined dataset
        combined_path = os.path.join(output_dir, "combined_full_dataset.csv")
        combined_df.to_csv(combined_path)
        
        print(f"✅ Combined dataset saved: {combined_path}")
        print(f"   Total samples: {len(combined_df)}")
        print(f"   Date range: {combined_df.index[0]} → {combined_df.index[-1]}")
        print(f"   Columns: {len(combined_df.columns)}")
        print()
        
        # Also save as parquet for faster loading
        parquet_path = os.path.join(output_dir, "combined_full_dataset.parquet")
        combined_df.to_parquet(parquet_path)
        print(f"✅ Also saved as: {parquet_path}")
        
        return combined_df
    else:
        print("❌ No data was fetched successfully")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch historical data from Cortex in chunks")
    parser.add_argument("--start", type=str, default="2024-01-01T00:00:00Z",
                       help="Start date (ISO format)")
    parser.add_argument("--end", type=str, default="2025-10-17T00:00:00Z",
                       help="End date (ISO format)")
    parser.add_argument("--chunk-months", type=int, default=3,
                       help="Chunk size in months (default: 3)")
    parser.add_argument("--step", type=str, default="15m",
                       help="Data resolution (default: 15m)")
    parser.add_argument("--output-dir", type=str, default="artifacts/historical_data",
                       help="Output directory for CSV files")
    
    args = parser.parse_args()
    
    # Fetch data
    df = fetch_data_in_chunks(
        start_date=args.start,
        end_date=args.end,
        chunk_months=args.chunk_months,
        step=args.step,
        output_dir=args.output_dir
    )
    
    if df is not None:
        print("\n" + "="*80)
        print("🎉 SUCCESS! Historical data fetched and combined.")
        print("="*80)
        print("\nTo train on this data, modify config.yaml:")
        print("  1. Comment out 'start' and 'end' in data section")
        print("  2. Add: use_cached_dataset: true")
        print("  3. Add: cached_dataset_path: 'artifacts/historical_data/combined_full_dataset.parquet'")
        print("\nThen run: python -m src.pipeline.train")

