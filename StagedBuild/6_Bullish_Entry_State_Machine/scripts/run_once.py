#!/usr/bin/env python3
"""Process one nowcaster prediction through the state machine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from config import load_config  # noqa: E402
from models import NowcasterBar  # noqa: E402
from state_machine import BullishTransitionStateMachine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prediction through the entry state machine")
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    parser.add_argument("--input", "-i", help="JSON file with Stage 5 /prediction payload")
    parser.add_argument("--in-position", action="store_true", help="Simulate in long position")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else project_root / "config.yaml"
    app_cfg = load_config(config_path)

    if args.input:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        data = json.load(sys.stdin)

    bar = NowcasterBar.from_prediction_json(data)
    sm = BullishTransitionStateMachine(app_cfg.state_machine)
    output = sm.process_bar(bar, in_position=args.in_position)
    print(json.dumps(output.to_dict(), indent=2))


if __name__ == "__main__":
    main()
