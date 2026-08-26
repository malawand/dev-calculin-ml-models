"""Load versioned YAML config with environment overrides. No secrets in files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return cfg


def env_or(cfg: dict[str, Any], env_key: str, default_key: str, default: Any = None) -> Any:
    env_name = cfg.get(env_key)
    if env_name:
        val = os.environ.get(str(env_name))
        if val is not None and val != "":
            return val
    if default_key in cfg and cfg[default_key] is not None:
        return cfg[default_key]
    return default


def cortex_url(cfg: dict[str, Any]) -> str:
    c = cfg["cortex"]
    return str(env_or(c, "url_env", "url_default", "http://10.1.20.60:9009/prometheus/api/v1/query_range"))


def clickhouse_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    c = cfg["clickhouse"]
    password = os.environ.get(str(c.get("password_env", "CLICKHOUSE_PASSWORD")), "")
    port = env_or(c, "port_env", "port_default", 8124)
    return {
        "host": str(env_or(c, "host_env", "host_default", "10.1.19.249")),
        "port": int(port),
        "username": str(env_or(c, "user_env", "user_default", "exporter")),
        "password": password,
        "table": str(env_or(c, "table_env", "table_default", "binance_ww.raw_ticks_binance")),
        "symbol": c.get("symbol", "BTCUSDT"),
    }
