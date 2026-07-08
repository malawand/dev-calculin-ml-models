# Stage 6 — Bullish Entry State Machine

Stage 6 sits **on top of** the Stage 5 nowcaster. It does not retrain models or fetch Prometheus directly (except via HTTP poll of Stage 5). It converts regime **probabilities** into a **structured entry setup** and optionally emits `ENTER_LONG`.

This is a **context / gating layer**, not a raw “buy when TRENDING_UP is highest” rule.

---

## What it detects

A specific bullish transition pattern:

```
NEUTRAL → CHOP_BASE → EXPANSION_ALERT → BULLISH_CONFIRMATION → LONG_ENTRY
```

| State | Meaning |
|-------|---------|
| `NEUTRAL` | No active setup |
| `CHOP_BASE` | CHOP dominated the lookback window |
| `EXPANSION_ALERT` | VOLATILE_EXPANSION prob spiked; setup timer starts |
| `BULLISH_CONFIRMATION` | TRENDING_UP overtaking TRENDING_DOWN |
| `LONG_ENTRY` | Transient: breakout + confidence filters passed → signal emitted |
| `IN_LONG` | External bot reports an open long |
| `COOLDOWN` | Post-signal cooldown before a new setup |

**State numeric values (Prometheus `btc_entry_sm_state_numeric`):**

| Value | State |
|-------|-------|
| 0 | NEUTRAL |
| 1 | CHOP_BASE |
| 2 | EXPANSION_ALERT |
| 3 | BULLISH_CONFIRMATION |
| 4 | LONG_ENTRY |
| 5 | IN_LONG |
| 6 | COOLDOWN |

**Nowcaster regime ints (Stage 5 `btc_regime_detector_regime_value`):** 0=CHOP, 1=TRENDING_UP, 2=TRENDING_DOWN, 3=VOLATILE_EXPANSION.

---

## Transition rules (summary)

1. **CHOP_BASE** — Over `chop_lookback_bars`, at least `required_chop_dominance_ratio` of bars have CHOP highest and ≥ `chop_dominance_threshold`.
2. **EXPANSION_ALERT** — Current vol-exp ≥ threshold and rise vs `volatile_expansion_lookback_bars` ago ≥ rise threshold.
3. **BULLISH_CONFIRMATION** — `prob_trending_up` ≥ threshold and spread vs down ≥ `trend_spread_threshold` (optional crossing mode in config).
4. **LONG_ENTRY** — Price breaks above prior-bar range high (+ buffer), confidence gap passes, not already in position (unless configured).
5. **Expiry** — After EXPANSION_ALERT, if confirmation/entry does not complete within `max_signal_age_bars`, reset to NEUTRAL.
6. **Cooldown** — After `ENTER_LONG`, wait `entry_cooldown_bars` before a new setup.

All range/breakout math uses **prior bars only** (no lookahead).

---

## Configuration

Edit [`config.yaml`](config.yaml). Defaults assume **15-minute bars** (same as Stage 5):

| Bars | Time @ 15m |
|------|------------|
| 4 | 1 hour |
| 8 | 2 hours |
| 12 | 3 hours |
| 96 | 24 hours |

Key thresholds live under `state_machine:` in config.

---

## Output format

Each bar produces:

```json
{
  "timestamp": "2026-05-30T21:51:36+00:00",
  "state": "BULLISH_CONFIRMATION",
  "state_numeric": 3,
  "action": "HOLD",
  "reason": "Awaiting price breakout above recent range high",
  "metadata": {
    "prob_chop": 0.21,
    "prob_volatile_expansion": 0.06,
    "prob_trending_up": 0.67,
    "prob_trending_down": 0.07,
    "chop_dominance_ratio": 0.75,
    "volatile_expansion_rise": 0.18,
    "trend_spread": 0.60,
    "recent_range_high": 73800.0,
    "breakout_level": 73873.8,
    "price_breakout": false,
    "confidence_gap": 0.45,
    "setup_age_bars": 2,
    "long_entry_signal": false
  }
}
```

Actions: `NO_TRADE`, `HOLD`, `ENTER_LONG` (`EXIT_LONG` reserved for future).

---

## Running

### Tests

```bash
cd StagedBuild/6_Bullish_Entry_State_Machine
python3 -m pytest tests/ -v
```

### Historical backtest

Scores Stage 3 parquet with Stage 4 classifier, replays through the state machine:

```bash
python3 scripts/backtest.py
python3 scripts/backtest.py --start 2026-02-16 --end 2026-03-01
```

Outputs: `reports/signals.parquet`, `reports/enter_long_events.json`.

### Single bar debug

```bash
curl -s http://localhost:8080/prediction | python3 scripts/run_once.py
```

Or from a saved JSON file:

```bash
python3 scripts/run_once.py --input sample_prediction.json
```

### Live (HTTP poll Stage 5)

Stage 5 must be running and reachable.

```bash
# Edit config.yaml nowcaster.url if needed
python3 src/live_service.py --config config.yaml
```

Endpoints:

| URL | Purpose |
|-----|---------|
| `GET :8082/health` | Liveness |
| `GET :8082/ready` | True after first poll |
| `GET :8082/signal` | Latest state machine output |
| `GET :9111/metrics` | Prometheus (`btc_entry_sm_*`) |

---

## Prometheus metrics

Prefix: `btc_entry_sm_*` (separate from Stage 5 `btc_regime_detector_*`).

Useful for Grafana:

- `btc_entry_sm_state_numeric` — current state
- `btc_entry_sm_chop_dominance_ratio`
- `btc_entry_sm_trend_spread`
- `btc_entry_sm_price_breakout`
- `btc_entry_sm_long_entry_signal`
- `btc_entry_sm_enter_long_total`

---

## Integration with bots

Recommended pattern:

```
nowcaster probabilities → Stage 6 state machine → allowed strategies / size
your alpha model       → entries and exits
```

Pass `in_position=True` to `process_bar()` (or `--in-position` on live service) when your bot holds a long to suppress duplicate entries.

Do **not** wire `ENTER_LONG` directly without your own execution, risk, and position checks.

---

## Files

| Path | Purpose |
|------|---------|
| `src/state_machine.py` | Core incremental state machine |
| `src/conditions.py` | Pure threshold checks |
| `src/models.py` | Types and I/O contracts |
| `src/config.py` | YAML config loader |
| `src/history.py` | Rolling bar buffer |
| `src/telemetry.py` | Prometheus exporter |
| `src/live_service.py` | HTTP poll loop + API |
| `scripts/backtest.py` | Historical replay |
| `scripts/run_once.py` | One-shot debug |

---

## Further reading

- [../5_Live_Service/WHAT_THIS_DOES.md](../5_Live_Service/WHAT_THIS_DOES.md) — nowcaster
- [../4_Classifier/WHAT_THIS_DOES.md](../4_Classifier/WHAT_THIS_DOES.md) — regime probabilities
- [../HOW_TO_RUN.md](../HOW_TO_RUN.md) — full pipeline
