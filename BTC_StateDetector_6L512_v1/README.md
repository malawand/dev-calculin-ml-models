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
    - If the location of where metrics are written need to be changed, feel free to modify write_prometheus_metrics.py @ line 11: METRICS_FILE = Path("./metrics/btc_ml_metrics.prom")  # Change path if needed
