#!/usr/bin/env python3
"""
Replay historical nowcaster outputs through the bullish entry state machine.

Option A — score Stage 3 parquet with Stage 4 classifier (default):
    python3 scripts/backtest.py

Option B — use precomputed predictions parquet with columns:
    prob_CHOP, prob_TRENDING_UP, prob_TRENDING_DOWN, prob_VOLATILE_EXPANSION,
    confidence (optional), plus price column or separate price parquet

Writes:
    reports/signals.parquet
    reports/enter_long_events.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
STAGE3 = PROJECT_ROOT.parent / "3_Momentum_State_Encoder"
STAGE4 = PROJECT_ROOT.parent / "4_Classifier"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(STAGE4))

from config import load_config  # noqa: E402
from models import Action, NowcasterBar  # noqa: E402
from state_machine import BullishTransitionStateMachine  # noqa: E402


DEFAULT_INPUT = STAGE3 / "btc_data_15m_mstate.parquet"
DEFAULT_PRICE_COL = "crypto_last_price"


def _bars_from_predictions(df: pd.DataFrame, price_col: str) -> list[NowcasterBar]:
    bars: list[NowcasterBar] = []
    for ts, row in df.sort_index().iterrows():
        if price_col not in row or pd.isna(row[price_col]):
            continue
        try:
            p_chop = float(row["prob_CHOP"])
            p_up = float(row["prob_TRENDING_UP"])
            p_down = float(row["prob_TRENDING_DOWN"])
            p_vol = float(row["prob_VOLATILE_EXPANSION"])
        except KeyError as exc:
            raise SystemExit(f"Predictions missing column: {exc}") from exc

        conf = (
            float(row["confidence"])
            if "confidence" in row and not pd.isna(row["confidence"])
            else max(p_chop, p_up, p_down, p_vol)
        )
        bars.append(
            NowcasterBar(
                timestamp=pd.Timestamp(ts).to_pydatetime(),
                price=float(row[price_col]),
                prob_chop=p_chop,
                prob_trending_up=p_up,
                prob_trending_down=p_down,
                prob_volatile_expansion=p_vol,
                confidence=conf,
            )
        )
    return bars


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    if args.predictions:
        return pd.read_parquet(args.predictions).sort_index()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    df = pd.read_parquet(input_path).sort_index()
    from predict import RegimeClassifier  # noqa: E402

    clf = RegimeClassifier(Path(args.artifacts))
    feature_cols = clf.feature_cols
    nan_mask = df[feature_cols].isna().any(axis=1)
    valid = df.loc[~nan_mask].copy()
    print(f"Scoring {len(valid):,} rows with Stage 4 classifier...")
    preds = clf.predict(valid)
    merged = valid.join(preds)
    if args.price_col in valid.columns:
        merged[args.price_col] = valid[args.price_col]
    elif "price" in valid.columns:
        merged[args.price_col] = valid["price"]
    else:
        for candidate in ("crypto_last_price", "close", "price"):
            if candidate in valid.columns:
                merged[args.price_col] = valid[candidate]
                break
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest bullish entry state machine")
    parser.add_argument("--config", "-c", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Stage 3 feature parquet")
    parser.add_argument("--predictions", help="Precomputed predictions parquet")
    parser.add_argument(
        "--artifacts",
        default=str(STAGE4 / "model_artifacts"),
        help="Stage 4 model artifacts dir",
    )
    parser.add_argument("--price-col", default=DEFAULT_PRICE_COL)
    parser.add_argument("--start", help="ISO start timestamp (inclusive)")
    parser.add_argument("--end", help="ISO end timestamp (inclusive)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
    args = parser.parse_args()

    app_cfg = load_config(args.config)
    df = load_predictions(args)

    if args.start:
        df = df.loc[pd.Timestamp(args.start, tz="UTC") :]
    if args.end:
        df = df.loc[: pd.Timestamp(args.end, tz="UTC")]

    if df.empty:
        raise SystemExit("No rows to backtest after filters.")

    bars = _bars_from_predictions(df, args.price_col)
    sm = BullishTransitionStateMachine(app_cfg.state_machine)

    records = []
    enter_events = []
    for bar in bars:
        out = sm.process_bar(bar)
        rec = out.to_dict()
        rec["price"] = bar.price
        records.append(rec)
        if out.action == Action.ENTER_LONG:
            enter_events.append(rec)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_path = out_dir / "signals.parquet"
    pd.DataFrame(records).to_parquet(signals_path, index=False)

    events_path = out_dir / "enter_long_events.json"
    events_path.write_text(json.dumps(enter_events, indent=2), encoding="utf-8")

    print(f"Processed {len(bars):,} bars")
    print(f"ENTER_LONG signals: {len(enter_events)}")
    print(f"Wrote {signals_path}")
    print(f"Wrote {events_path}")


if __name__ == "__main__":
    main()
