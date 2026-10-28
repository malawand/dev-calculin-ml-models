import subprocess
import sys
import os

def run_script(script_name):
    """Runs a python script and streams its output to the terminal."""
    print(f"\n{'='*50}")
    print(f"🚀 STARTING: {script_name}")
    print(f"{'='*50}\n")
    
    if not os.path.exists(script_name):
        print(f"❌ ERROR: Cannot find {script_name}")
        sys.exit(1)

    try:
        # We use subprocess.run to execute the script in the same terminal
        # check=True means it will throw an error if the script fails (e.g. syntax error)
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {script_name} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n⚠️ WARNING: Process interrupted by user while running {script_name}.")
        sys.exit(1)

def main():
    print("🌟 Starting Binance Backfill Pipeline...\n")
    
    # 1. Export the gaps
    run_script("extract_gaps.py")
    
    # 2. Check if there are actually gaps to process
    if not os.path.exists("pending_gaps.json"):
        print("\n🎉 No pending_gaps.json found. Pipeline complete.")
        return
        
    # Optional: Read the json to see if it's empty
    import json
    try:
        with open("pending_gaps.json", "r") as f:
            gaps = json.load(f)
            if not gaps:
                print("\n🎉 pending_gaps.json is empty. Pipeline complete.")
                return
    except Exception as e:
         print(f"\n⚠️ Could not read pending_gaps.json: {e}")
    
    # 3. Calculate the missing data
    run_script("recreate_24h_ticker_for_gap.py")
    
    # 4. Upload to ClickHouse
    run_script("upload_data.py")
    
    print("\n✅ All scripts executed successfully!")

if __name__ == "__main__":
    main()
