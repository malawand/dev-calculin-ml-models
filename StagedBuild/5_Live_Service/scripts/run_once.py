#!/usr/bin/env python3
"""Run a single prediction cycle locally (no HTTP server)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import yaml  # noqa: E402
from predict import LivePredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one live regime prediction")
    parser.add_argument("--config", "-c", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    predictor = LivePredictor.from_config(config, base_dir=PROJECT_ROOT)
    output = predictor.predict_once()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
