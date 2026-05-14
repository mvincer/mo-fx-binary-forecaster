"""Turn classifier probabilities into directional strategy returns and risk analytics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

TradeMode = Literal["always", "threshold"]


def holdout_oos_parts(dset: dict[str, Any], preds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Single 50/50 split → one pseudo fold for the same backtest utilities as walk-forward."""
    if dset.get("error") or "forward_r_test" not in dset:
        return []
    ix = dset["X_test"].index
    return [
        {
            "dates_te": pd.DatetimeIndex(pd.to_datetime(ix)),
            "y_te": dset["y_test"].to_numpy(),
            "forward_r_te": dset["forward_r_test"].to_numpy(dtype=float),
            "preds": preds,
        },
    ]


def stitched_oos_frame(
    oos_parts: list[dict[str, Any]],
    model_name: str,
) -> pd.DataFrame | None:
    """Concatenate walk-forward test segments for one model (chronological)."""
    chunks: list[pd.DataFrame] = []
    for b in oos_parts:
        pr = b["preds"].get(model_name)
        if pr is None or pr.get("proba_te") is None:
            continue
        proba = np.asarray(pr["proba_te"])
        pred = np.asarray(pr["pred_te"]).ravel()
        dt = b["dates_te"]
        d = pd.DataFrame(
            {
                "date": pd.to_datetime(dt),
                "y_true": np.asarray(b["y_te"]).ravel(),
                "forward_r": np.asarray(b["forward_r_te"], dtype=float).ravel(),
                "pred": pred,
            },
        )
        for j in range(proba.shape[1]):
            d[f"p_{j}"] = proba[:, j]
        chunks.append(d)
    if not chunks:
        return None
    out = pd.concat(chunks, axis=0, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    # Non-overlapping folds should not duplicate dates; if they do, keep last fold’s row.
    dup = out["date"].duplicated(keep=False)
    if dup.any():
        out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def stitched_oos_summary(oos_parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Earliest / latest calendar date and row count implied by stitched OOS blocks (for UI)."""
    ts: list[pd.Timestamp] = []
    n = 0
    for b in oos_parts:
        d = b.get("dates_te")
        if d is None or len(d) == 0:
            continue
        dd = pd.DatetimeIndex(pd.to_datetime(d))
        ts.extend(dd.tolist())
        n += len(dd)
    if not ts:
        return {
            "min_date": None,
            "max_date": None,
            "n_rows": 0,
            "n_unique_dates": 0,
        }
    ser = pd.DatetimeIndex(ts)
    u = ser.unique().sort_values()
    return {
        "min_date": ser.min(),
        "max_date": ser.max(),
        "n_rows": int(n),
        "n_unique_dates": int(len(u)),
    }


def strategy_returns_binary(
    frame: pd.DataFrame,
    *,
    mode: TradeMode,
    prob_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Binary: position +1 (long) if class 1 wins else -1 (short); threshold mode requires max prob."""
    p0 = frame["p_0"].to_numpy(dtype=float)
    p1 = frame["p_1"].to_numpy(dtype=float)
    r = frame["forward_r"].to_numpy(dtype=float)
    max_p = np.maximum(p0, p1)
    side = np.where(p1 >= p0, 1.0, -1.0)
    if mode == "threshold":
        trade = max_p >= prob_threshold
        side = np.where(trade, side, 0.0)
    pos = pd.Series(side, index=frame.index)
    return pd.Series(side * r, index=frame.index), pos


def strategy_returns_ternary(
    frame: pd.DataFrame,
    *,
    mode: TradeMode,
    prob_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Ternary: flat if predicted mid (class 1); long class 2 (+r), short class 0 (-r)."""
    p0 = frame["p_0"].to_numpy(dtype=float)
    p1 = frame["p_1"].to_numpy(dtype=float)
    p2 = frame["p_2"].to_numpy(dtype=float)
    r = frame["forward_r"].to_numpy(dtype=float)
    stk = np.stack([p0, p1, p2], axis=1)
    max_p = stk.max(axis=1)
    cls = np.argmax(stk, axis=1)
    pos = np.zeros(len(r))
    pos[cls == 2] = 1.0
    pos[cls == 0] = -1.0
    pos[cls == 1] = 0.0
    if mode == "threshold":
        ok = max_p >= prob_threshold
        pos = np.where(ok, pos, 0.0)
    ps = pd.Series(pos, index=frame.index)
    return pd.Series(pos * r, index=frame.index), ps


def performance_stats(
    strat_r: pd.Series,
    positions: pd.Series | None = None,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Annualized return, Sharpe, Sortino, max drawdown, hit rate on periods with non-flat position."""
    x = pd.to_numeric(strat_r, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return {
            "n_days": float(len(x)),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "pct_traded": float("nan"),
        }

    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    neg = x[x < 0]
    downside = float(neg.std(ddof=1)) if len(neg) > 1 else float("nan")

    eq = (1.0 + x).cumprod()
    peak = eq.cummax()
    dd = float(((eq / peak) - 1.0).min())

    if positions is not None:
        al = positions.reindex(x.index).fillna(0.0)
        traded_mask = al.abs() > 1e-12
        if traded_mask.any():
            sub = x[traded_mask]
            hit = float((sub > 0).mean())
            pct_tr = float(traded_mask.mean())
        else:
            hit = float("nan")
            pct_tr = 0.0
    else:
        nz = x != 0
        hit = float((x[nz] > 0).mean()) if nz.any() else float("nan")
        pct_tr = float(nz.mean())

    ann_ret = (1.0 + mu) ** periods_per_year - 1.0
    ann_vol = sd * np.sqrt(periods_per_year)
    sharpe = (mu / sd) * np.sqrt(periods_per_year) if sd > 1e-15 else float("nan")
    sortino = (mu / downside) * np.sqrt(periods_per_year) if downside and downside > 1e-15 else float("nan")

    return {
        "n_days": float(len(x)),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": dd,
        "hit_rate": hit,
        "pct_traded": pct_tr,
    }


def oos_rolling_block_stats(
    strat_r: pd.Series,
    positions: pd.Series | None,
    *,
    chunk_size: int = 100,
    periods_per_year: float = 252.0,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """
    Split **chronological** OOS strategy returns into consecutive blocks of ``chunk_size`` rows
    (each row = one OOS prediction / P&L period). Computes :func:`performance_stats` on each block.

    Returns (per_block_stats, meta) where meta includes ``n_oos_rows``, ``n_chunks``,
    ``chunk_size_requested``, and ``note`` (e.g. fewer than 100 OOS rows).
    """
    x = pd.to_numeric(strat_r, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pos = (
        positions.reindex(x.index).fillna(0.0)
        if positions is not None
        else pd.Series(0.0, index=x.index)
    )
    m = x.notna()
    x = x[m]
    pos = pos[m]
    n = int(len(x))
    meta: dict[str, Any] = {
        "n_oos_rows": n,
        "chunk_size_requested": int(chunk_size),
        "n_chunks": 0,
        "note": "",
    }
    if n == 0:
        return [], meta
    if n < chunk_size:
        meta["note"] = f"Only {n} OOS rows (< {chunk_size}); one block uses all rows."
    blocks: list[dict[str, float]] = []
    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        st = performance_stats(x.iloc[sl], positions=pos.iloc[sl], periods_per_year=periods_per_year)
        st["block_i"] = float(len(blocks))
        st["block_n"] = float(sl.stop - sl.start)
        blocks.append(st)
    meta["n_chunks"] = len(blocks)
    return blocks, meta


def summarize_rolling_blocks(
    blocks: list[dict[str, float]],
    *,
    metrics: tuple[str, ...] = (
        "sharpe",
        "ann_return",
        "sortino",
        "max_drawdown",
        "hit_rate",
        "pct_traded",
        "ann_vol",
    ),
) -> dict[str, dict[str, float]]:
    """Mean, std, min, max of each metric across non-NaN block values."""
    out: dict[str, dict[str, float]] = {}
    if not blocks:
        return out
    for k in metrics:
        vals = []
        for b in blocks:
            if k not in b:
                continue
            v = b[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            vals.append(float(v))
        if not vals:
            out[k] = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n_blocks": 0.0}
            continue
        a = np.asarray(vals, dtype=float)
        std = float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0
        out[k] = {
            "mean": float(np.nanmean(a)),
            "std": std,
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
            "n_blocks": float(len(vals)),
        }
    return out


def flatten_rolling_summary(prefix: str, summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """Wide dict for CSV columns ``{prefix}sharpe_mean``, etc."""
    flat: dict[str, float] = {}
    for metric, d in summary.items():
        for stat in ("mean", "std", "min", "max", "n_blocks"):
            if stat in d:
                flat[f"{prefix}{metric}_{stat}"] = float(d[stat])
    return flat


def equity_curve(strat_r: pd.Series, dates: pd.Series | None = None) -> pd.DataFrame:
    r = pd.to_numeric(strat_r, errors="coerce")
    eq = (1.0 + r).cumprod()
    out = pd.DataFrame({"strategy_return": r, "equity": eq})
    if dates is not None:
        out.insert(0, "date", pd.to_datetime(dates))
    return out
