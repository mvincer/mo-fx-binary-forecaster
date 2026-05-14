"""Direction targets from forward simple returns (same definition as Currencies ``forward_bar_returns`` h≥1)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_window_excursions(high: pd.Series, low: pd.Series, close: pd.Series, lead: int) -> pd.DataFrame:
    """
    Per row ``t``, look at daily bars **t+1 … t+lead** (inclusive of ``t+lead``).

    Returns a frame aligned to ``close.index`` with:

    - ``hi_exc``: ``max(High) / Close[t] - 1`` over that window
    - ``lo_exc``: ``min(Low) / Close[t] - 1`` over that window
    - ``close_ret``: ``Close[t+lead] / Close[t] - 1`` (same horizon as lead-day close-to-close)
    """
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").astype(float)
    c = pd.to_numeric(close, errors="coerce").astype(float)
    hv = h.values
    lv = l.values
    cv = c.values
    n = len(c)
    hi_exc = np.full(n, np.nan)
    lo_exc = np.full(n, np.nan)
    close_ret = np.full(n, np.nan)
    L = int(lead)
    for i in range(n):
        if i + L >= n:
            break
        sl = slice(i + 1, i + L + 1)
        hh = hv[sl]
        ll = lv[sl]
        if not (np.all(np.isfinite(hh)) and np.all(np.isfinite(ll)) and np.isfinite(cv[i])):
            continue
        den = cv[i]
        if abs(den) < 1e-18:
            continue
        hi_exc[i] = float(np.nanmax(hh)) / den - 1.0
        lo_exc[i] = float(np.nanmin(ll)) / den - 1.0
        close_ret[i] = cv[i + L] / den - 1.0 if np.isfinite(cv[i + L]) else np.nan

    idx = c.index
    return pd.DataFrame(
        {"hi_exc": hi_exc, "lo_exc": lo_exc, "close_ret": close_ret},
        index=idx,
    )


def binary_labels_high_low(exc: pd.DataFrame, *, tie_break_close: bool = True) -> pd.Series:
    """
    Path-based binary label using forward window highs/lows vs entry **close**.

    - **Up (1)** if upside excursion dominates: ``hi_exc ≥ abs(lo_exc)`` and ``hi_exc > 0``.
    - **Down (0)** if downside dominates: ``abs(lo_exc) > hi_exc`` and ``lo_exc < 0``.
    - **Tie** (equal dominance): use sign of ``close_ret`` if ``tie_break_close``, else NaN.
    """
    hi = exc["hi_exc"].to_numpy(dtype=float)
    lo = exc["lo_exc"].to_numpy(dtype=float)
    cr = exc["close_ret"].to_numpy(dtype=float)
    y = np.full(len(hi), np.nan)
    for i in range(len(hi)):
        if not (np.isfinite(hi[i]) and np.isfinite(lo[i])):
            continue
        hix = hi[i]
        lox = lo[i]
        ml = abs(lox)
        if hix >= ml and hix > 0:
            y[i] = 1.0
        elif ml > hix and lox < 0:
            y[i] = 0.0
        elif tie_break_close and np.isfinite(cr[i]):
            if cr[i] > 0:
                y[i] = 1.0
            elif cr[i] < 0:
                y[i] = 0.0
    return pd.Series(y, index=exc.index)


def ternary_labels_high_low(exc: pd.DataFrame, *, band: float = 0.005) -> pd.Series:
    """
    Ternary from path excursions vs entry close:

    - **Mid (1)** if ``hi_exc ≤ band`` and ``lo_exc ≥ -band``.
    - **Up (2)** / **Down (0)** when that corridor is breached; both breached → larger excursion wins.
    """
    hi = exc["hi_exc"].to_numpy(dtype=float)
    lo = exc["lo_exc"].to_numpy(dtype=float)
    y = np.full(len(hi), np.nan)
    b = float(band)
    for i in range(len(hi)):
        if not (np.isfinite(hi[i]) and np.isfinite(lo[i])):
            continue
        hix = hi[i]
        lox = lo[i]
        if (hix <= b) and (lox >= -b):
            y[i] = 1.0
            continue
        up_hit = hix > b
        dn_hit = lox < -b
        if up_hit and not dn_hit:
            y[i] = 2.0
        elif dn_hit and not up_hit:
            y[i] = 0.0
        elif up_hit and dn_hit:
            y[i] = 2.0 if hix >= abs(lox) else 0.0
        elif hix > b:
            y[i] = 2.0
        elif lox < -b:
            y[i] = 0.0
        else:
            y[i] = 1.0
    return pd.Series(y, index=exc.index)


def forward_simple_return(close: pd.Series, lead: int) -> pd.Series:
    """``Close[t+lead]/Close[t]-1`` for ``lead >= 1``."""
    s = close.astype(float)
    return s.shift(-lead) / s - 1.0


def binary_labels(r: pd.Series) -> pd.Series:
    """1 = up, 0 = down; NaN where ``r`` is NaN or exactly 0."""
    y = pd.Series(np.nan, index=r.index, dtype=float)
    y[r > 0] = 1.0
    y[r < 0] = 0.0
    return y


def ternary_labels(r: pd.Series, *, band: float = 0.005) -> pd.Series:
    """
    0 = ``r < -band``, 1 = mid band, 2 = ``r > band`` (``band`` in fraction, 0.005 = 0.5%).
    """
    y = pd.Series(np.nan, index=r.index, dtype=float)
    y[r < -band] = 0.0
    y[(r >= -band) & (r <= band)] = 1.0
    y[r > band] = 2.0
    return y
