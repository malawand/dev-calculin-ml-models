#!/usr/bin/env python3
"""
Live regime detection service.

Runs a loop: fetch Prometheus → encode → classify → expose results.

Endpoints (on api_port, default 8080):
  GET /health      — liveness probe
  GET /ready       — readiness (true after first successful prediction)
  GET /prediction  — latest prediction JSON

Prometheus metrics (on metrics_port, default 9109):
  GET /metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from predict import LivePredictor  # noqa: E402
from telemetry import RegimeTelemetry  # noqa: E402

logger = logging.getLogger(__name__)

_latest_prediction: dict[str, Any] | None = None
_prediction_lock = threading.Lock()


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        if self.path == "/ready":
            with _prediction_lock:
                ready = _latest_prediction is not None
            self._send_json(200 if ready else 503, {"ready": ready})
            return

        if self.path == "/prediction":
            with _prediction_lock:
                if _latest_prediction is None:
                    self._send_json(503, {"error": "No prediction yet"})
                    return
                self._send_json(200, _latest_prediction)
            return

        self._send_json(404, {"error": "Not found"})


def start_api_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("API listening on :%s (/health, /ready, /prediction)", port)
    return server


def run_loop(predictor: LivePredictor, telemetry: RegimeTelemetry, interval: int) -> None:
    global _latest_prediction

    while True:
        try:
            output = predictor.predict_once()
            with _prediction_lock:
                _latest_prediction = output
            telemetry.emit(output)
            logger.info(
                "regime=%s confidence=%.3f price=%s",
                output["regime"],
                output["confidence"],
                output.get("price"),
            )
            print(json.dumps(output, indent=2))
        except Exception as exc:
            telemetry.record_error()
            logger.exception("Prediction cycle failed: %s", exc)
        time.sleep(interval)


def main(config_path: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if config_path is None:
        config_path = str(PROJECT_ROOT / "config.yaml")
    config = load_config(Path(config_path))

    api_port = int(config.get("service", {}).get("api_port", 8080))
    interval = int(config.get("service", {}).get("poll_interval_seconds", 900))

    predictor = LivePredictor.from_config(config, base_dir=PROJECT_ROOT)
    telemetry = RegimeTelemetry(config)
    telemetry.start()
    start_api_server(api_port)

    logger.info("Starting prediction loop (interval=%ss)", interval)
    run_loop(predictor, telemetry, interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC regime live detection service")
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
