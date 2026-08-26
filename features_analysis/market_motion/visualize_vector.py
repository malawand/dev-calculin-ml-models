"""Full-history visualization of cortex_market_motion_v1. No scoring or outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from . import TRANSITION_RAW_UP_AFTER_TROUGH
from .transition import EVENT_NONE, EVENT_PEAK, EVENT_TROUGH

DISCLAIMER = (
    "PROVISIONAL POC — temporary Cortex-derived vector. "
    "Visualization only. No outcomes, future returns, or trading labels."
)

HORIZONS = ("2h", "4h", "8h", "12h", "24h", "48h", "7d")
FIRST_ALL_HORIZON_VALID = pd.Timestamp("2023-08-12 03:00:00", tz="UTC")
HTML_STEP_HOURS = 2

CMAP_T = ListedColormap(["#2e7d4f", "#c0392b"]).with_extremes(bad="#9aa0a6")
CMAP_DIV = plt.get_cmap("RdBu_r").copy().with_extremes(bad="#9aa0a6")
CMAP_AGE = plt.get_cmap("viridis").copy().with_extremes(bad="#9aa0a6")

NEEDED = [
    "time_bucket",
    "btc_close_at_t",
    "row_vector_valid",
    "cross_n_horizons_up",
    "cross_n_horizons_valid",
]
for _h in HORIZONS:
    NEEDED.extend(
        [
            f"{_h}_derivative_raw",
            f"{_h}_position_z",
            f"{_h}_velocity",
            f"{_h}_acceleration",
            f"{_h}_transition_state_raw",
            f"{_h}_hours_since_confirmation",
            f"{_h}_feature_valid",
            f"{_h}_transition_event",
        ]
    )


def load_vector(path: Path) -> pd.DataFrame:
    wanted = set(NEEDED)
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True)
    df = df.sort_values("time_bucket").reset_index(drop=True)
    valid = df["row_vector_valid"].fillna(False).astype(bool)
    if not bool(valid.any()):
        raise ValueError("no all-horizon-valid rows in vector")
    start = df.loc[valid, "time_bucket"].iloc[0]
    return df.loc[df["time_bucket"] >= start].reset_index(drop=True)


def downsample_for_html(df: pd.DataFrame, hours: int = HTML_STEP_HOURS) -> pd.DataFrame:
    """Keep every Nth hourly row. T is not backdated; confirmation still appears on the next kept bar."""
    if hours <= 1:
        return df.copy()
    return df.iloc[::hours].reset_index(drop=True)


def robust_sym_scale(values: np.ndarray, *, clip: float = 4.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.full_like(x, np.nan, dtype=float)
    mad = float(np.median(np.abs(finite - np.median(finite))))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.nanstd(finite)) or 1.0
    return np.clip(x / scale, -clip, clip)


def masked_state(df: pd.DataFrame, horizon: str) -> np.ndarray:
    t = pd.to_numeric(df[f"{horizon}_transition_state_raw"], errors="coerce").to_numpy(dtype=float)
    ok = df[f"{horizon}_feature_valid"].fillna(False).astype(bool).to_numpy()
    return np.where(ok, t, np.nan)


def event_masks(df: pd.DataFrame, horizon: str) -> tuple[np.ndarray, np.ndarray]:
    ev = df[f"{horizon}_transition_event"].fillna(EVENT_NONE).astype(str)
    return ev.eq(EVENT_PEAK).to_numpy(), ev.eq(EVENT_TROUGH).to_numpy()


def heatmap_matrix(df: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows = horizons (2h at top). Color matrix plus unscaled hover matrix."""
    n = len(df)
    color = np.full((len(HORIZONS), n), np.nan)
    raw = np.full((len(HORIZONS), n), np.nan)
    for i, h in enumerate(HORIZONS):
        ok = df[f"{h}_feature_valid"].fillna(False).astype(bool).to_numpy()
        if kind == "T":
            raw[i] = masked_state(df, h)
            color[i] = raw[i]
        elif kind == "D":
            raw[i] = pd.to_numeric(df[f"{h}_derivative_raw"], errors="coerce").to_numpy(dtype=float)
            color[i] = np.where(ok, robust_sym_scale(np.where(ok, raw[i], np.nan)), np.nan)
        elif kind == "Z":
            raw[i] = pd.to_numeric(df[f"{h}_position_z"], errors="coerce").to_numpy(dtype=float)
            color[i] = np.where(ok, np.clip(raw[i], -8.0, 8.0), np.nan)
        elif kind == "V":
            raw[i] = pd.to_numeric(df[f"{h}_velocity"], errors="coerce").to_numpy(dtype=float)
            color[i] = np.where(ok, robust_sym_scale(np.where(ok, raw[i], np.nan)), np.nan)
        elif kind == "A":
            raw[i] = pd.to_numeric(df[f"{h}_acceleration"], errors="coerce").to_numpy(dtype=float)
            color[i] = np.where(ok, robust_sym_scale(np.where(ok, raw[i], np.nan)), np.nan)
        elif kind == "age":
            raw[i] = pd.to_numeric(df[f"{h}_hours_since_confirmation"], errors="coerce").to_numpy(dtype=float)
            color[i] = np.where(ok, np.log1p(np.clip(raw[i], 0, None)), np.nan)
        else:
            raise ValueError(kind)
    return color, raw


def describe_row(row: pd.Series) -> str:
    """Plain-language description of one hour. Not a forecast."""
    ts = pd.Timestamp(row["time_bucket"])
    px = row.get("btc_close_at_t")
    n_up = row.get("cross_n_horizons_up")
    n_valid = row.get("cross_n_horizons_valid")
    n_up_i = int(n_up) if pd.notna(n_up) else 0
    n_valid_i = int(n_valid) if pd.notna(n_valid) else 0
    lines = [f"At this hour ({ts.strftime('%Y-%m-%d %H:%M UTC')}):", ""]
    if pd.notna(px):
        lines.append(f"BTC close at t is {float(px):,.2f}.")
    lines.append(f"{n_up_i} of {n_valid_i} valid horizons are upward after a trough.")
    for h in HORIZONS:
        ok = bool(row.get(f"{h}_feature_valid"))
        if not ok:
            lines.append(f"The {h} curve is invalid or unknown this hour.")
            continue
        t = row.get(f"{h}_transition_state_raw")
        z = row.get(f"{h}_position_z")
        v = row.get(f"{h}_velocity")
        if pd.isna(t):
            lines.append(f"The {h} curve has no confirmed transition yet.")
            continue
        after = (
            "upward after a trough"
            if float(t) == TRANSITION_RAW_UP_AFTER_TROUGH
            else "downward after a peak"
        )
        z_s = ""
        if pd.notna(z) and abs(float(z)) >= 2.0:
            z_s = " unusually elevated" if float(z) > 0 else " unusually depressed"
        motion = ""
        if pd.notna(v):
            av = abs(float(v))
            if float(v) > 0:
                motion = " and rising" + (" more slowly" if av < 0.25 else "")
            elif float(v) < 0:
                motion = " and falling" + (" more slowly" if av < 0.25 else "")
        lines.append(f"The {h} curve is {after}{z_s}{motion}.")
    lines.append("This is a description of the vector, not a forecast or trading recommendation.")
    return "\n".join(lines)


def _extent(times: pd.DatetimeIndex) -> list[float]:
    t0 = mdates.date2num(pd.Timestamp(times[0]).to_pydatetime())
    t1 = mdates.date2num(pd.Timestamp(times[-1]).to_pydatetime())
    return [t0, t1, len(HORIZONS) - 0.5, -0.5]


def _format_time_axis(ax: Any) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())


def write_overview_png(df: pd.DataFrame, dest: Path) -> None:
    times = pd.DatetimeIndex(df["time_bucket"])
    ext = _extent(times)
    fig = plt.figure(figsize=(22, 18), facecolor="white")
    gs = GridSpec(8, 1, height_ratios=[2.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.4], hspace=0.12, figure=fig)
    ax_px = fig.add_subplot(gs[0])
    px = pd.to_numeric(df["btc_close_at_t"], errors="coerce")
    ax_px.plot(times, px, color="#1a1a1a", lw=0.6)
    ax_px.set_yscale("log")
    ax_px.set_ylabel("BTC (log USD)")
    ax_px.set_title(
        "Cortex market-motion vector v1 · prod_a_preferred · full usable history\n" + DISCLAIMER,
        fontsize=11,
        loc="left",
    )
    ax_px.grid(True, alpha=0.25)
    specs = [
        ("T", "T  green=0 up after trough · red=1 down after peak · grey=unknown", CMAP_T, False),
        ("D", "D  per-horizon robust color; raw magnitudes are not comparable across rows", CMAP_DIV, True),
        ("Z", "Z  common −8 to +8", CMAP_DIV, True),
        ("V", "V  per-horizon robust symmetric color (rising vs falling)", CMAP_DIV, True),
        ("A", "A  per-horizon robust symmetric color (strengthening vs weakening)", CMAP_DIV, True),
        ("age", "Hours since confirmation  color=log1p", CMAP_AGE, False),
    ]
    axes = [ax_px]
    for i, (kind, title, cmap, diverging) in enumerate(specs, start=1):
        ax = fig.add_subplot(gs[i], sharex=ax_px)
        color, _raw = heatmap_matrix(df, kind)
        kw: dict[str, Any] = dict(aspect="auto", interpolation="nearest", origin="upper", extent=ext, cmap=cmap)
        if kind == "T":
            kw.update(vmin=0, vmax=1)
        elif kind == "Z":
            kw["norm"] = TwoSlopeNorm(vcenter=0.0, vmin=-8.0, vmax=8.0)
        elif kind == "age":
            finite = color[np.isfinite(color)]
            kw.update(vmin=0, vmax=float(np.nanpercentile(finite, 99)) if finite.size else 5.0)
        elif diverging:
            kw["norm"] = TwoSlopeNorm(vcenter=0.0, vmin=-4.0, vmax=4.0)
        ax.imshow(np.ma.masked_invalid(color), **kw)
        ax.set_yticks(np.arange(len(HORIZONS)))
        ax.set_yticklabels(HORIZONS, fontsize=8)
        ax.set_ylabel(title, fontsize=8)
        axes.append(ax)
    ax_c = fig.add_subplot(gs[7], sharex=ax_px)
    ax_c.plot(times, df["cross_n_horizons_up"], color="#2e7d4f", lw=0.7, label="horizons upward after trough")
    ax_c.plot(times, df["cross_n_horizons_valid"], color="#4a4a4a", lw=0.7, label="horizons valid")
    ax_c.set_ylim(-0.2, 7.4)
    ax_c.set_ylabel("Cross-horizon count")
    ax_c.legend(loc="upper left", fontsize=8, frameon=False)
    ax_c.set_xlabel("UTC")
    for ax in axes:
        _format_time_axis(ax)
        ax.tick_params(labelbottom=False)
    _format_time_axis(ax_c)
    ax_c.tick_params(labelbottom=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_horizon_png(df: pd.DataFrame, horizon: str, dest: Path) -> None:
    times = pd.DatetimeIndex(df["time_bucket"])
    px = pd.to_numeric(df["btc_close_at_t"], errors="coerce")
    d = pd.to_numeric(df[f"{horizon}_derivative_raw"], errors="coerce")
    z = pd.to_numeric(df[f"{horizon}_position_z"], errors="coerce")
    v = pd.to_numeric(df[f"{horizon}_velocity"], errors="coerce")
    a = pd.to_numeric(df[f"{horizon}_acceleration"], errors="coerce")
    age = pd.to_numeric(df[f"{horizon}_hours_since_confirmation"], errors="coerce")
    state = masked_state(df, horizon)
    peaks, troughs = event_masks(df, horizon)
    fig, axes = plt.subplots(8, 1, figsize=(20, 16), sharex=True, facecolor="white")
    fig.subplots_adjust(hspace=0.08)
    axes[0].plot(times, px, color="#1a1a1a", lw=0.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BTC (log USD)")
    axes[0].set_title(
        f"{horizon} Cortex D · confirmation-time markers (not hindsight extrema)\n{DISCLAIMER}",
        loc="left",
        fontsize=10,
    )
    axes[1].plot(times, d, color="#1f4e79", lw=0.5)
    axes[1].set_ylabel(f"{horizon} D")
    axes[1].axhline(0, color="#888", lw=0.4)
    axes[2].plot(times, d, color="#1f4e79", lw=0.45)
    d_np = d.to_numpy()
    axes[2].scatter(times[peaks], d_np[peaks], marker="v", s=14, c="#c0392b", label="confirmed peak", zorder=3)
    axes[2].scatter(times[troughs], d_np[troughs], marker="^", s=14, c="#2e7d4f", label="confirmed trough", zorder=3)
    axes[2].legend(loc="upper left", fontsize=8, frameon=False)
    axes[2].set_ylabel(f"{horizon} D + events")
    axes[2].axhline(0, color="#888", lw=0.4)
    t_img = np.ma.masked_invalid(state.reshape(1, -1))
    ext = [
        mdates.date2num(pd.Timestamp(times[0]).to_pydatetime()),
        mdates.date2num(pd.Timestamp(times[-1]).to_pydatetime()),
        0,
        1,
    ]
    axes[3].imshow(t_img, aspect="auto", interpolation="nearest", origin="upper", extent=ext, cmap=CMAP_T, vmin=0, vmax=1)
    axes[3].set_ylabel("T strip")
    axes[3].set_yticks([])
    axes[4].plot(times, z, color="#b35c00", lw=0.5)
    axes[4].axhline(0, color="#888", lw=0.4)
    axes[4].set_ylabel("Z")
    axes[5].plot(times, v, color="#6a1b9a", lw=0.5)
    axes[5].axhline(0, color="#888", lw=0.4)
    axes[5].set_ylabel("V")
    axes[6].plot(times, a, color="#1565c0", lw=0.5)
    axes[6].axhline(0, color="#888", lw=0.4)
    axes[6].set_ylabel("A")
    axes[7].plot(times, age, color="#333", lw=0.5, label="hours since confirmation")
    invalid = ~df[f"{horizon}_feature_valid"].fillna(False).astype(bool).to_numpy()
    if invalid.any():
        ymax = float(np.nanmax(age.to_numpy())) if np.isfinite(age).any() else 1.0
        axes[7].fill_between(times, 0, ymax, where=invalid, color="#9aa0a6", alpha=0.35, label="invalid")
    axes[7].legend(loc="upper left", fontsize=8, frameon=False)
    axes[7].set_ylabel("Hours / validity")
    axes[7].set_xlabel("UTC")
    _format_time_axis(axes[7])
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.savefig(dest, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_overview_html(df: pd.DataFrame, dest: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    times = pd.DatetimeIndex(df["time_bucket"])
    px = pd.to_numeric(df["btc_close_at_t"], errors="coerce")
    titles = [
        "BTC close at t (log USD)",
        "T  green=0 up after trough · red=1 down after peak · grey=unknown/invalid",
        "D  color is per-horizon robust scale; hover shows unscaled D",
        "Z  common −8…+8",
        "V  per-horizon robust symmetric color",
        "A  per-horizon robust symmetric color",
        "Hours since confirmation  color=log1p; hover=hours",
        "Cross-horizon counts (0–7)",
    ]
    fig = make_subplots(
        rows=8,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.18, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.16],
        subplot_titles=titles,
    )
    custom = np.column_stack(
        [
            pd.to_numeric(df["cross_n_horizons_up"], errors="coerce"),
            pd.to_numeric(df["cross_n_horizons_valid"], errors="coerce"),
        ]
    )
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=px,
            mode="lines",
            name="BTC",
            line=dict(color="#111", width=1),
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M} UTC<br>BTC %{y:.2f}"
                "<br>upward horizons %{customdata[0]:.0f}"
                "<br>valid horizons %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(type="log", title_text="USD", row=1, col=1)

    t_cs = [[0.0, "#2e7d4f"], [0.5, "#c0392b"], [1.0, "#9aa0a6"]]
    specs = [
        (2, "T", t_cs, 0.0, 2.0, "T"),
        (3, "D", "RdBu_r", -4.0, 4.0, "D"),
        (4, "Z", "RdBu_r", -8.0, 8.0, "Z"),
        (5, "V", "RdBu_r", -4.0, 4.0, "V"),
        (6, "A", "RdBu_r", -4.0, 4.0, "A"),
        (7, "age", "Viridis", 0.0, None, "hours"),
    ]
    for row, kind, cs, zmin, zmax, label in specs:
        color, raw = heatmap_matrix(df, kind)
        if kind == "T":
            color = np.where(np.isfinite(color), color, 2.0)
        if kind == "age":
            finite = color[np.isfinite(color)]
            zmax = float(np.nanpercentile(finite, 99)) if finite.size else 5.0
        fig.add_trace(
            go.Heatmap(
                z=color,
                x=times,
                y=list(HORIZONS),
                colorscale=cs,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(len=0.09, thickness=12, y=1.02 - row * 0.115),
                customdata=np.expand_dims(raw, axis=2),
                name=kind,
                hoverongaps=False,
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M} UTC<br>%{y} "
                    + label
                    + " %{customdata[0]:.6g}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=pd.to_numeric(df["cross_n_horizons_up"], errors="coerce"),
            mode="lines",
            name="n upward after trough",
            line=dict(color="#2e7d4f", width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>upward %{y:.0f}<extra></extra>",
        ),
        row=8,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=pd.to_numeric(df["cross_n_horizons_valid"], errors="coerce"),
            mode="lines",
            name="n valid",
            line=dict(color="#444", width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>valid %{y:.0f}<extra></extra>",
        ),
        row=8,
        col=1,
    )
    fig.update_yaxes(range=[-0.2, 7.4], title_text="count", row=8, col=1)
    fig.update_layout(
        height=2200,
        showlegend=True,
        legend=dict(orientation="h", y=1.02),
        hovermode="x",
        spikedistance=-1,
        title=dict(text=DISCLAIMER, x=0.01, xanchor="left", font=dict(size=12)),
        margin=dict(t=80, l=70, r=40, b=40),
    )
    fig.update_xaxes(showspikes=True, spikemode="across+marker", spikesnap="cursor", rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=True, row=8, col=1)
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, config={"scrollZoom": True})
    dest.write_text(html.replace("</body>", _snapshot_script(df) + "</body>"), encoding="utf-8")


def _json_num(v: Any, nd: int | None = None) -> float | int | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return None
    x = float(v)
    if nd is None:
        return x
    return round(x, nd)


def _snapshot_script(df: pd.DataFrame) -> str:
    payload: dict[str, Any] = {
        "t": df["time_bucket"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "px": [_json_num(v, 2) for v in df["btc_close_at_t"]],
        "up": [_json_num(v, 0) for v in df["cross_n_horizons_up"]],
        "nv": [_json_num(v, 0) for v in df["cross_n_horizons_valid"]],
    }
    for h in HORIZONS:
        payload[h] = {
            "T": [_json_num(v, 0) for v in df[f"{h}_transition_state_raw"]],
            "Z": [_json_num(v, 4) for v in df[f"{h}_position_z"]],
            "V": [_json_num(v, 4) for v in df[f"{h}_velocity"]],
            "A": [_json_num(v, 6) for v in df[f"{h}_acceleration"]],
            "D": [_json_num(v, 6) for v in df[f"{h}_derivative_raw"]],
            "age": [_json_num(v, 2) for v in df[f"{h}_hours_since_confirmation"]],
            "ok": [bool(v) for v in df[f"{h}_feature_valid"].fillna(False)],
        }
    blob = json.dumps(payload, separators=(",", ":"))
    return f"""
<div id="mmv-snapshot" style="position:sticky;bottom:0;background:#f7f7f5;border-top:1px solid #ccc;padding:12px 16px;font:14px/1.45 ui-sans-serif,system-ui,sans-serif;max-height:28vh;overflow:auto;white-space:pre-wrap;z-index:20;">
Hover or click an hour. The text below describes the vector at that time. It is not a forecast.
</div>
<script>
window.MMV = {blob};
window.MMV_MS = window.MMV.t.map(function(s) {{ return Date.parse(s); }});
function mmvIndex(x) {{
  const ms = Date.parse(x);
  const a = window.MMV_MS;
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (a[mid] < ms) lo = mid + 1; else hi = mid;
  }}
  if (lo > 0 && Math.abs(a[lo-1] - ms) < Math.abs(a[lo] - ms)) return lo - 1;
  return lo;
}}
function mmvDescribe(i) {{
  const d = window.MMV;
  if (!d || i < 0 || i >= d.t.length) return "";
  const hs = ["2h","4h","8h","12h","24h","48h","7d"];
  const lines = ["At this hour (" + d.t[i] + "):", ""];
  if (d.px[i] != null) lines.push("BTC close at t is " + Number(d.px[i]).toLocaleString(undefined, {{maximumFractionDigits: 2}}) + ".");
  lines.push((d.up[i] ?? 0) + " of " + (d.nv[i] ?? 0) + " valid horizons are upward after a trough.");
  for (const h of hs) {{
    const r = d[h];
    if (!r.ok[i]) {{ lines.push("The " + h + " curve is invalid or unknown this hour."); continue; }}
    const T = r.T[i];
    if (T == null) {{ lines.push("The " + h + " curve has no confirmed transition yet."); continue; }}
    let after = T === 0 ? "upward after a trough" : "downward after a peak";
    let zs = "";
    if (r.Z[i] != null && Math.abs(r.Z[i]) >= 2) zs = r.Z[i] > 0 ? " unusually elevated" : " unusually depressed";
    let motion = "";
    if (r.V[i] != null) {{
      const av = Math.abs(r.V[i]);
      if (r.V[i] > 0) motion = " and rising" + (av < 0.25 ? " more slowly" : "");
      else if (r.V[i] < 0) motion = " and falling" + (av < 0.25 ? " more slowly" : "");
    }}
    lines.push("The " + h + " curve is " + after + zs + motion + ".");
    lines.push("  D=" + r.D[i] + "  Z=" + r.Z[i] + "  V=" + r.V[i] + "  A=" + r.A[i] + "  hours_since_confirmation=" + r.age[i]);
  }}
  lines.push("This is a description of the vector, not a forecast or trading recommendation.");
  return lines.join("\\n");
}}
function mmvBind() {{
  const gd = document.querySelector(".plotly-graph-div");
  const box = document.getElementById("mmv-snapshot");
  if (!gd || !box || gd._mmvBound) return;
  gd._mmvBound = true;
  gd.on("plotly_hover", ev => {{
    if (!ev.points || !ev.points.length) return;
    box.textContent = mmvDescribe(mmvIndex(ev.points[0].x));
  }});
  gd.on("plotly_click", ev => {{
    if (!ev.points || !ev.points.length) return;
    box.textContent = mmvDescribe(mmvIndex(ev.points[0].x));
  }});
}}
document.addEventListener("DOMContentLoaded", mmvBind);
setTimeout(mmvBind, 400);
</script>
"""


def write_readme(df: pd.DataFrame, dest: Path, *, html_hours: int) -> None:
    last = df.iloc[-1]
    first = df.iloc[0]
    dest.write_text(
        f"""# Cortex market-motion vector visualization

{DISCLAIMER}

**Vector:** `cortex_market_motion_v1` · `prod_a_preferred_v1`  
**Period:** {first['time_bucket']} → {last['time_bucket']}  
**Rows:** {len(df):,} hourly bars, from the first `row_vector_valid` hour (expected {FIRST_ALL_HORIZON_VALID})

This folder is visualization only. It does not retrain, score, or tune the detector or analogue matcher. No analogue outcomes, CRPS, future returns, oracle extrema, or buy/sell labels are plotted.

## Files

- `market_motion_overview.html` — interactive dashboard with a shared UTC axis, zoom/pan, a bottom rangeslider, a vertical spike cursor, and a sticky plain-language snapshot. Heatmaps in the HTML are downsampled to **{html_hours}-hour** last-value bins so a browser can render the stack; state still changes at confirmation time, never earlier. PNGs below are hourly.
- `market_motion_overview.png` — static full-history hourly stack.
- `horizon_*.png` — one detailed hourly stack per Cortex horizon.

## Components

For each of `2h, 4h, 8h, 12h, 24h, 48h, 7d`:

| Field | Meaning |
|---|---|
| `derivative_raw` | Cortex D at that horizon (raw units are not comparable across horizons). |
| `position_z` | Causal z-score of D. |
| `velocity` | Rate of change of D (rising vs falling). |
| `acceleration` | Rate of change of velocity (strengthening vs weakening). |
| `transition_state_raw` | Persistent confirmed turn of **D**. |
| `hours_since_confirmation` | Age of the live T state. |
| `feature_valid` | Horizon usable this hour. Shown so gaps are visible; not a model-distance input. |

Cross-horizon:

- `cross_n_horizons_up` — count of valid horizons with T=`0` (0–7).
- `cross_n_horizons_valid` — count of valid horizons (0–7).

Price is `btc_close_at_t` on a **logarithmic** scale.

## Transition convention (`0` / `1`)

This is **not** sign(D).

| Value | Meaning | Color |
|---|---|---|
| `0` | Upward after a confirmed trough | Green |
| `1` | Downward after a confirmed peak | Red |
| null / `feature_valid=false` | Unknown or invalid | Grey |

## Color meanings (heatmaps)

- **T:** categorical green / red / grey as above. Invalid hours stay grey even if a stale `0`/`1` is sitting in the CSV.
- **D, V, A:** diverging blue (negative) → near-zero → orange-red (positive). Color is a **per-horizon robust MAD scale** clipped to about ±4 so a 7d wiggle is visible next to a 2h wiggle. Hover on the HTML shows the **unscaled** value.
- **Z:** common scale **−8 to +8**, same diverging map.
- **Hours since confirmation:** `log1p` color so a 7d state that is weeks old does not wash out a 2h state that is two hours old. Hover shows the actual hour count.

## Why D is not on one color bar

Raw Cortex derivative magnitudes differ by horizon (2h is much larger than 7d). A shared color bar would make long horizons look flat. Each D row is colored with its own robust scale; the number on hover is the real D.

## Confirmation-time placement

State changes and peak/trough **markers sit on the confirmation bar**, the same hour `transition_event` is `PEAK_CONFIRMED` or `TROUGH_CONFIRMED`. They are **not** moved back to `candidate_extremum_time`. Earlier bars are never rewritten after a later confirmation (`docs/TIMESTAMP_RULE.md`).

## Snapshot text

The HTML sticky panel translates the hovered or clicked hour into language from that row only (T, Z, V, D, A, age, validity, counts). It is a description of the vector, **not** a forecast or trading recommendation.

## Last hour (data, not a forecast)

```
{describe_row(last)}
```
""",
        encoding="utf-8",
    )


def canvas_series(df: pd.DataFrame) -> dict[str, Any]:
    monthly = df.set_index("time_bucket")["btc_close_at_t"].resample("MS").last().dropna()
    last = df.iloc[-1]
    horizons = []
    for h in HORIZONS:
        t = last.get(f"{h}_transition_state_raw")
        ok = bool(last.get(f"{h}_feature_valid"))
        if not ok or pd.isna(t):
            state = "unknown/invalid"
        elif float(t) == TRANSITION_RAW_UP_AFTER_TROUGH:
            state = "0 up after trough"
        else:
            state = "1 down after peak"
        horizons.append(
            {
                "horizon": h,
                "state": state,
                "valid": ok,
                "D": _json_num(last.get(f"{h}_derivative_raw"), 6),
                "Z": _json_num(last.get(f"{h}_position_z"), 3),
                "V": _json_num(last.get(f"{h}_velocity"), 3),
                "A": _json_num(last.get(f"{h}_acceleration"), 5),
                "age": _json_num(last.get(f"{h}_hours_since_confirmation"), 1),
            }
        )
    return {
        "n_rows": int(len(df)),
        "start_utc": str(df["time_bucket"].iloc[0]),
        "end_utc": str(df["time_bucket"].iloc[-1]),
        "last_px": _json_num(last.get("btc_close_at_t"), 2),
        "last_up": _json_num(last.get("cross_n_horizons_up"), 0),
        "last_valid": _json_num(last.get("cross_n_horizons_valid"), 0),
        "snapshot": describe_row(last),
        "horizons": horizons,
        "monthly": {
            "labels": [ts.strftime("%Y-%m") for ts in monthly.index],
            "close": [round(float(v), 2) for v in monthly.to_numpy()],
        },
    }


def run_visualization(vector_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_vector(vector_path)
    write_overview_png(df, out_dir / "market_motion_overview.png")
    for h in HORIZONS:
        write_horizon_png(df, h, out_dir / f"horizon_{h}.png")
    html_df = downsample_for_html(df)
    write_overview_html(html_df, out_dir / "market_motion_overview.html")
    write_readme(df, out_dir / "README.md", html_hours=HTML_STEP_HOURS)
    extra = canvas_series(df)
    return {
        "disclaimer": DISCLAIMER,
        "n_rows": extra["n_rows"],
        "start_utc": extra["start_utc"],
        "end_utc": extra["end_utc"],
        "html_rows": int(len(html_df)),
        "html_step_hours": HTML_STEP_HOURS,
        "out_dir": str(out_dir),
        "last_snapshot": extra["snapshot"],
        "canvas": extra,
    }
