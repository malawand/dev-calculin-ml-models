"""Hourly BTCUSDT close. Password from CLICKHOUSE_PASSWORD only."""

from __future__ import annotations

import pandas as pd


def fetch_hourly_closes(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    if not password:
        raise RuntimeError(
            "CLICKHOUSE_PASSWORD is not set. Export it or copy .env.example to .env. "
            "Do not put the password in source."
        )
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host=host, username=username, password=password, port=int(port)
    )
    start_d = pd.Timestamp(start, tz="UTC").strftime("%Y-%m-%d")
    end_d = pd.Timestamp(end, tz="UTC").strftime("%Y-%m-%d")
    safe_symbol = symbol.replace("'", "")
    q = f"""
    SELECT
        toStartOfInterval(event_time, INTERVAL 1 HOUR) AS time_bucket,
        argMax(last_price, event_time) AS close_price
    FROM {table}
    WHERE symbol = '{safe_symbol}'
      AND event_time >= '{start_d}'
      AND event_time <= '{end_d}'
    GROUP BY time_bucket
    ORDER BY time_bucket
    """
    df = client.query_df(q)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True)
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    return df
