#!/usr/bin/env python3
"""
Live entry state machine service.

Polls Stage 5 nowcaster GET /prediction every poll_interval_seconds,
feeds each bar through the bullish transition state machine, and exposes:
  GET :api_port/health
  GET :api_port/ready
  GET :api_port/signal
  GET :metrics_port/metrics
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

import requests

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from models import NowcasterBar  # noqa: E402
from state_machine import BullishTransitionStateMachine  # noqa: E402
from telemetry import EntryStateMachineTelemetry  # noqa: E402

logger = logging.getLogger(__name__)

_latest_signal: dict[str, Any] | None = None
_signal_lock = threading.Lock()


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
            with _signal_lock:
                ready = _latest_signal is not None
            self._send_json(200 if ready else 503, {"ready": ready})
            return

        if self.path == "/signal":
            with _signal_lock:
                if _latest_signal is None:
                    self._send_json(503, {"error": "No signal yet"})
                    return
                self._send_json(200, _latest_signal)
            return

        self._send_json(404, {"error": "Not found"})


def start_api_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("API listening on :%s (/health, /ready, /signal)", port)
    return server


def fetch_prediction(url: str) -> dict[str, Any]:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.json()


def run_loop(
    sm: BullishTransitionStateMachine,
    telemetry: EntryStateMachineTelemetry,
    nowcaster_url: str,
    interval: int,
    in_position: bool,
) -> None:
    global _latest_signal

    while True:
        try:
            data = fetch_prediction(nowcaster_url)
            bar = NowcasterBar.from_prediction_json(data)
            output = sm.process_bar(bar, in_position=in_position)
            payload = output.to_dict()
            with _signal_lock:
                _latest_signal = payload
            telemetry.emit(output)
            logger.info(
                "state=%s action=%s reason=%s",
                output.state.value,
                output.action.value,
                output.reason,
            )
            if output.action.value == "ENTER_LONG":
                logger.warning("ENTER_LONG signal emitted at %s", output.timestamp)
            print(json.dumps(payload, indent=2))
        except Exception as exc:
            telemetry.record_error()
            logger.exception("Poll cycle failed: %s", exc)
        time.sleep(interval)


def main(config_path: str | None = None, in_position: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if config_path is None:
        config_path = str(PROJECT_ROOT / "config.yaml")
    raw_path = Path(config_path)
    app_cfg = load_config(raw_path)

    with open(raw_path, encoding="utf-8") as f:
        raw_yaml = __import__("yaml").safe_load(f)

    sm = BullishTransitionStateMachine(app_cfg.state_machine)
    telemetry = EntryStateMachineTelemetry(raw_yaml or {})
    telemetry.start()
    start_api_server(app_cfg.service.api_port)

    logger.info(
        "Starting entry SM loop (poll=%ss, nowcaster=%s)",
        app_cfg.nowcaster.poll_interval_seconds,
        app_cfg.nowcaster.url,
    )
    run_loop(
        sm,
        telemetry,
        app_cfg.nowcaster.url,
        app_cfg.nowcaster.poll_interval_seconds,
        in_position,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bullish entry state machine live service")
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--in-position",
        action="store_true",
        help="Simulate already holding a long (blocks reentry)",
    )
    args = parser.parse_args()
    main(args.config, in_position=args.in_position)
