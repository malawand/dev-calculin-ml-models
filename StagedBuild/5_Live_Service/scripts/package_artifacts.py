#!/usr/bin/env python3
"""Copy Stage 4 classifier artifacts into Stage 5 artifacts/."""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE4_ARTIFACTS = PROJECT_ROOT.parent / "4_Classifier" / "model_artifacts"
DEST = PROJECT_ROOT / "artifacts"

FILES = ["model.txt", "feature_cols.json", "class_weights.json", "metrics.json"]


def main() -> None:
    if not STAGE4_ARTIFACTS.exists():
        raise SystemExit(f"Stage 4 artifacts not found: {STAGE4_ARTIFACTS}")

    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = STAGE4_ARTIFACTS / name
        if src.exists():
            shutil.copy2(src, DEST / name)
            print(f"Copied {name}")
        else:
            print(f"Skipped (missing): {name}")

    print(f"\nClassifier artifacts ready in {DEST}/")
    print("Next: python3 scripts/export_encoder_artifacts.py")


if __name__ == "__main__":
    main()
