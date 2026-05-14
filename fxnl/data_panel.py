"""Cross-pair feature matrix: primary pair fundamentals + tech, other pairs' ``{pid}_*`` columns only."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from fxnl.paths import patch_currencies_sys_path
from fxnl.targets import (
    binary_labels,
    binary_labels_high_low,
    forward_simple_return,
    forward_window_excursions,
    ternary_labels,
    ternary_labels_high_low,
)
from fxnl.vol_features import add_garch_vol_columns

logger = logging.getLogger(__name__)


def _assemble_one(instrument: str, period: str | None, study_bars: int | None):
    from fxnl.data_repo import assemble_merged_from_repo, use_repo_data  # noqa: PLC0415

    if use_repo_data():
        m, pid, cc = assemble_merged_from_repo(instrument, period=period, study_bars=study_bars)
        if m is not None and not m.empty:
            return m, pid, cc
    patch_currencies_sys_path()
    from src.merged_data import assemble_merged_panel  # noqa: PLC0415

    return assemble_merged_panel(instrument, period=period, study_bars=study_bars)


def _primary_high_low_close(
    primary_instrument: str,
    period: str | None,
    study_bars: int | None,
    index: pd.Index,
) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """Align FXCM **High**, **Low**, **Close** to ``index`` (same calendar as merged panel)."""
    from fxnl.data_repo import load_primary_fx_ohlc, use_repo_data  # noqa: PLC0415

    if use_repo_data():
        ohlc = load_primary_fx_ohlc(primary_instrument, period, study_bars, index)
        if ohlc is not None:
            return ohlc

    patch_currencies_sys_path()
    from src.fxcm_loader import load_instrument_cached  # noqa: PLC0415
    from src.merged_data import trim_period, _strip_tz_index  # noqa: PLC0415

    raw = load_instrument_cached(primary_instrument)
    if raw.empty or "High" not in raw.columns or "Low" not in raw.columns:
        return None
    raw = _strip_tz_index(raw)
    raw = trim_period(raw, period)
    if study_bars is not None and int(study_bars) > 0:
        raw = raw.tail(int(study_bars))
    hi = raw["High"].astype(float).reindex(index)
    lo = raw["Low"].astype(float).reindex(index)
    cl = raw["Close"].astype(float).reindex(index)
    return hi, lo, cl


def prepare_direction_frame(
    primary_instrument: str,
    *,
    context_instruments: list[str],
    period: str | None,
    study_bars: int | None,
    lead: int,
    target_mode: str,
    exclude_all_close: bool = True,
    ternary_band: float = 0.005,
    include_garch_vol: bool = True,
    target_return_basis: str = "close",
    keep_unlabeled_tail: bool = False,
) -> dict[str, Any]:
    """
    Build aligned ``X``, ``y``, forward returns ``r``, and calendar index — **no** train/test split.

    When ``include_garch_vol`` is True, appends ``feat_garch_*`` columns from rolling GARCH / realized vol.

    ``target_return_basis``: ``\"close\"`` = label from **close-to-close** net change; ``\"high_low\"`` =
    label from **max High / min Low** vs entry close over the next ``lead`` bars (path-consistent),
    while ``forward_r`` remains **close-to-close** for P&amp;L backtests.

    ``keep_unlabeled_tail`` keeps rows with usable features even when ``y`` / ``forward_r`` are NaN
    (the latest ``lead`` rows). Use this only for live forecasting; training/evaluation callers should
    keep the default ``False``.
    """
    if lead < 1:
        raise ValueError("lead must be >= 1")

    tb = str(target_return_basis or "close").strip().lower()

    merged_p, pid_p, close_col = _assemble_one(primary_instrument, period, study_bars)
    if merged_p.empty:
        return {"error": "empty_primary_panel"}

    primary_close = merged_p[close_col].astype(float)

    X = merged_p.copy()
    for ins in context_instruments:
        if ins == primary_instrument:
            continue
        m, pid_o, _ = _assemble_one(ins, period, study_bars)
        if m.empty:
            logger.warning("Skipping empty panel for context %s", ins)
            continue
        take = [c for c in m.columns if str(c).startswith(f"{pid_o}_")]
        if not take:
            continue
        sub = m[take].reindex(X.index)
        X = X.join(sub, how="left")
        for c in sub.columns:
            if c in X.columns:
                X[c] = X[c].ffill(limit=5)

    if exclude_all_close:
        close_cols = [c for c in X.columns if str(c).endswith("_close")]
        X = X.drop(columns=close_cols, errors="ignore")

    if include_garch_vol:
        try:
            X = add_garch_vol_columns(X, primary_close.reindex(X.index))
        except Exception as e:
            logger.warning("GARCH vol features skipped: %s", e)

    X = X.replace([np.inf, -np.inf], np.nan)
    min_frac = 0.05 if len(X) < 400 else 0.12
    feat_ok = X.columns[X.notna().mean() >= min_frac]
    X = X[feat_ok]
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        return {"error": "no_feature_columns_after_filters", "X": X, "y": pd.Series(dtype=int)}

    X = X.sort_index()
    pc = primary_close.reindex(X.index).astype(float)
    r = forward_simple_return(pc, lead)
    hi_exc = None
    lo_exc = None
    if tb in ("high_low", "hl", "path"):
        ohlc = _primary_high_low_close(primary_instrument, period, study_bars, X.index)
        if ohlc is None:
            return {"error": "high_low_target_requires_valid_FXCM_High_Low_Close_cache"}
        high_s, low_s, close_s = ohlc
        exc = forward_window_excursions(high_s, low_s, close_s, lead)
        hi_exc = exc["hi_exc"]
        lo_exc = exc["lo_exc"]
        if target_mode == "binary":
            y = binary_labels_high_low(exc, tie_break_close=True)
        elif target_mode == "ternary":
            y = ternary_labels_high_low(exc, band=ternary_band)
        else:
            raise ValueError("target_mode must be 'binary' or 'ternary'")
    else:
        if target_mode == "binary":
            y = binary_labels(r)
        elif target_mode == "ternary":
            y = ternary_labels(r, band=ternary_band)
        else:
            raise ValueError("target_mode must be 'binary' or 'ternary'")

    feature_use = X.notna().sum(axis=1) > 0 if X.shape[1] > 0 else pd.Series(False, index=X.index)
    label_use = y.notna() & r.notna()
    use = feature_use if keep_unlabeled_tail else (label_use & feature_use)
    X = X.loc[use]
    y = y.loc[use]
    if not keep_unlabeled_tail:
        y = y.astype(int)
    r = r.loc[use]
    if hi_exc is not None:
        hi_exc = hi_exc.loc[use]
        lo_exc = lo_exc.loc[use]
    dates = pd.DatetimeIndex(pd.to_datetime(X.index))

    n_labeled = int((y.notna() & r.notna()).sum())
    if n_labeled < 80:
        err_ret: dict[str, Any] = {
            "error": (
                f"too_few_labeled_rows:{n_labeled} — try a longer `period`, smaller cross-pair set, "
                f"or ensure FXCM/FRED cache covers this window."
            ),
            "X": X,
            "y": y,
            "forward_r": r,
            "dates": dates,
            "target_return_basis": tb,
        }
        if hi_exc is not None:
            err_ret["hi_exc"] = hi_exc
            err_ret["lo_exc"] = lo_exc
        return err_ret

    out_ok: dict[str, Any] = {
        "X": X,
        "y": y,
        "forward_r": r,
        "close": pc.loc[X.index],
        "dates": dates,
        "close_col": close_col,
        "primary_id": pid_p,
        "lead": lead,
        "target_mode": target_mode,
        "n_total": len(X),
        "n_labeled": n_labeled,
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "target_return_basis": tb,
    }
    if hi_exc is not None:
        out_ok["hi_exc"] = hi_exc
        out_ok["lo_exc"] = lo_exc
    return out_ok


def build_direction_dataset(
    primary_instrument: str,
    *,
    context_instruments: list[str],
    period: str | None,
    study_bars: int | None,
    lead: int,
    target_mode: str,
    exclude_all_close: bool = True,
    ternary_band: float = 0.005,
    include_garch_vol: bool = True,
    target_return_basis: str = "close",
) -> dict[str, Any]:
    """
    Chronological 50% / 50% train-test split on rows (in-sample / out-of-sample).

    ``target_mode``: ``\"binary\"`` or ``\"ternary\"``.
    """
    base = prepare_direction_frame(
        primary_instrument,
        context_instruments=context_instruments,
        period=period,
        study_bars=study_bars,
        lead=lead,
        target_mode=target_mode,
        exclude_all_close=exclude_all_close,
        ternary_band=ternary_band,
        include_garch_vol=include_garch_vol,
        target_return_basis=target_return_basis,
        keep_unlabeled_tail=False,
    )
    if base.get("error"):
        return base

    X = base["X"]
    y = base["y"]
    n = len(X)
    k = n // 2
    X_train, X_test = X.iloc[:k], X.iloc[k:]
    y_train, y_test = y.iloc[:k], y.iloc[k:]
    forward_r = base["forward_r"].loc[X.index]

    out = dict(base)
    out["X_train"] = X_train
    out["X_test"] = X_test
    out["y_train"] = y_train
    out["y_test"] = y_test
    out["forward_r"] = forward_r
    out["forward_r_train"] = forward_r.iloc[:k]
    out["forward_r_test"] = forward_r.iloc[k:]
    out["n_train"] = len(X_train)
    out["n_test"] = len(X_test)
    out["n_features"] = X.shape[1]
    return out
