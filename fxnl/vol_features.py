"""Rolling GARCH(1,1)-style volatility features without lookahead (fit on past returns only)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rolling_garch_ann_vol_pct(close: pd.Series, *, fit_window: int = 500, refit_every: int = 40) -> pd.Series:
    """
    One-step conditional volatility scale (percentage-return units), aligned to **return** dates.

    Refits every ``refit_every`` observations on the prior ``fit_window`` pct returns (×100).
    Falls back to EWMA if ``arch`` is missing or fits fail.
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    r = (c.pct_change() * 100.0).dropna()

    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch not installed — using EWMA vol proxy instead of GARCH")
        return _ewma_vol_proxy(r, halflife=60).reindex(c.index)

    rv = r.values
    min_fit = 100
    cached_vol = np.nan
    last_fit_at = -1
    vals_out = np.full(len(r), np.nan)

    for pos in range(min_fit, len(rv)):
        if last_fit_at < 0 or (pos - last_fit_at) >= refit_every:
            seg = rv[max(0, pos - fit_window) : pos]
            seg = seg[np.isfinite(seg)]
            if len(seg) < min_fit:
                continue
            try:
                am = arch_model(seg, vol="Garch", p=1, q=1, rescale=True)
                res = am.fit(disp="off", options={"maxiter": 80})
                fv = res.forecast(horizon=1, reindex=False)
                vari = float(np.asarray(fv.variance.iloc[-1]).ravel()[-1])
                cached_vol = float(np.sqrt(max(vari, 1e-16)))
                last_fit_at = pos
            except Exception:
                continue
        if np.isfinite(cached_vol):
            vals_out[pos] = cached_vol

    out_r = pd.Series(vals_out, index=r.index)
    sig = out_r.reindex(c.index).ffill()
    return sig


def _ewma_vol_proxy(r_pct: pd.Series, *, halflife: int = 60) -> pd.Series:
    return r_pct.pow(2).ewm(halflife=halflife, adjust=False).mean().pow(0.5)


def add_garch_vol_columns(X: pd.DataFrame, close_for_vol: pd.Series, *, prefix: str = "feat_garch_") -> pd.DataFrame:
    """Append GARCH cond. vol + short-horizon realized vol (past-only)."""
    sig = rolling_garch_ann_vol_pct(close_for_vol.reindex(X.index).ffill())
    X2 = X.copy()
    X2[f"{prefix}cond_sigma_pct"] = sig.reindex(X.index)
    r1 = close_for_vol.pct_change() * 100.0
    X2[f"{prefix}realized_vol20_pct"] = r1.rolling(20, min_periods=10).std().reindex(X.index)
    return X2
