# dev-calculin-ml-models

BTC_StateDetector_6L512_v1
     └─ Asset
         └─ Purpose
                └─ Layers & Size
                        └─ Version

To run this model, simply start the virtual environment with the following command:
source btc_direction_predictor/venv/bin/activate
python3 monitor_combined_realtime.py

This will start to run and promql metrics will be written to the metrics/ directory.
    - This should be written on to the PVC and scraped by the analytics prometheus scrape job in crypto-analytics