"""Run manifests: checksums, git sha, config hashes. Required by Gate 1."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_manifest(
    dest: Path,
    *,
    run_id: str,
    repo: Path,
    config_path: Path,
    config: dict[str, Any],
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_text = json.dumps(config, sort_keys=True, default=str).encode()
    man: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(repo),
        "config_path": str(config_path),
        "feature_configuration_hash": sha256_bytes(cfg_text),
        "timezone": config.get("timezone", "UTC"),
        "asset": config.get("asset"),
        "sampling_frequency": config.get("sampling_frequency"),
        "software_environment": {
            "python": os.sys.version.split()[0],
        },
        "inputs": {},
        "outputs": {},
        "row_counts": {},
    }
    for name, path in inputs.items():
        if path.exists():
            man["inputs"][name] = {"path": str(path), "sha256": sha256_file(path)}
    for name, path in outputs.items():
        if path.exists():
            rec = {"path": str(path), "sha256": sha256_file(path)}
            if path.suffix == ".csv":
                rec["rows"] = int(sum(1 for _ in open(path, encoding="utf-8")) - 1)
                man["row_counts"][name] = rec["rows"]
            man["outputs"][name] = rec
    if extra:
        man.update(extra)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(man, indent=2, default=str) + "\n", encoding="utf-8")
    return man


def csv_checksum(df: pd.DataFrame) -> str:
    raw = df.to_csv(index=False).encode()
    return sha256_bytes(raw)


def feature_configuration_hash(config: dict[str, Any]) -> str:
    cfg_text = json.dumps(config, sort_keys=True, default=str).encode()
    return sha256_bytes(cfg_text)
