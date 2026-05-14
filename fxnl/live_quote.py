"""Synthesize today's FX OHLC bar so the panel can forecast **tomorrow**.

Pre-collected daily parquets often end **before** the current session. When
``FXNL_INJECT_LIVE_QUOTE`` is set, :func:`append_live_today_bar` extends/updates
the last row so ``daily_binary_forecast_report`` sees **today** as
``latest_feature_date`` and ``--align-to-next-bar`` targets the **next** session.

**Sources (tried in order):**

1. **Yahoo** — intraday aggregate (1m / 5m / 15m / 1h), then **daily** carry-forward
   (synthetic NY-today bar from last close when needed).
2. **FXCM public D1 candledata** — last daily **BidClose** for the current (or prior) year.
3. **Frankfurter** (ECB) — ``latest?from=BASE&to=QUOTE`` spot.
4. **open.er-api.com** — free ``v6/latest/{BASE}`` cross table.

All fallbacks build a **single OHLC row** on the NY calendar **anchor** so the merged
panel always has a usable "today" close when any source responds.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_HTTP_UA = {"User-Agent": "fxnl-live-quote/1.1 (pandas; research)"}

YAHOO_FX_TICKERS: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X",
    "GBP/JPY": "GBPJPY=X",
}


def instrument_from_pid(pid: str) -> str:
    """Convert ``EUR_USD`` → ``EUR/USD`` (FX_PAIR_INSTRUMENTS form)."""
    return pid.replace("_", "/")


def _base_quote_codes(instrument: str) -> Optional[tuple[str, str]]:
    s = instrument.strip().upper().replace(" ", "")
    if "/" in s:
        a, b = s.split("/", 1)
        if len(a) == 3 and len(b) == 3 and a.isalpha() and b.isalpha():
            return a, b
    if len(s) == 6 and s.isalpha():
        return s[:3], s[3:]
    return None


def _synthetic_anchor_bar(anchor: pd.Timestamp, close: float) -> pd.DataFrame:
    c = float(close)
    return pd.DataFrame(
        {"Open": [c], "High": [c], "Low": [c], "Close": [c], "Volume": [0.0]},
        index=pd.DatetimeIndex([pd.Timestamp(anchor).normalize()]),
    )


def _fxcm_d1_last_close(instrument: str) -> Optional[float]:
    """Last **BidClose** from FXCM public yearly D1 CSV (gzip)."""
    sym = instrument.replace("/", "").replace(" ", "").strip().upper()
    if len(sym) != 6 or not sym.isalpha():
        return None
    year_now = datetime.now(timezone.utc).year
    for year in (year_now, year_now - 1):
        url = f"https://candledata.fxcorporate.com/D1/{sym}/{year}.csv.gz"
        req = urllib.request.Request(url, headers=_HTTP_UA)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = gzip.decompress(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            logger.debug("FXCM candledata HTTP %s %s: %s", sym, year, e)
            continue
        except Exception as e:
            logger.debug("FXCM candledata read %s %s: %s", sym, year, e)
            continue
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            logger.debug("FXCM candledata parse %s: %s", sym, e)
            continue
        cols = {str(c).strip(): c for c in df.columns}
        bc = cols.get("BidClose") or cols.get("Close")
        if bc is None or df.empty:
            continue
        c = pd.to_numeric(df[bc], errors="coerce").dropna()
        if c.empty:
            continue
        v = float(c.iloc[-1])
        if v > 0:
            logger.info("FXCM D1 last close for %s (%s): %.5f", instrument, year, v)
            return v
    return None


def _frankfurter_mid(base: str, quote: str) -> Optional[float]:
    try:
        url = f"https://api.frankfurter.app/latest?from={base}&to={quote}"
        req = urllib.request.Request(url, headers=_HTTP_UA)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.debug("Frankfurter %s/%s: %s", base, quote, e)
        return None
    rates = data.get("rates") or {}
    v = rates.get(quote)
    if v is None:
        v = rates.get(quote.upper())
    try:
        f = float(v)
        if f > 0:
            logger.info("Frankfurter spot %s/%s = %.5f", base, quote, f)
            return f
    except (TypeError, ValueError):
        pass
    return None


def _open_er_api_mid(base: str, quote: str) -> Optional[float]:
    try:
        url = f"https://open.er-api.com/v6/latest/{base}"
        req = urllib.request.Request(url, headers=_HTTP_UA)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.debug("open.er-api %s/%s: %s", base, quote, e)
        return None
    rates = data.get("rates") or {}
    v = rates.get(quote) if quote in rates else rates.get(quote.upper())
    try:
        f = float(v)
        if f > 0:
            logger.info("open.er-api spot %s/%s = %.5f", base, quote, f)
            return f
    except (TypeError, ValueError):
        pass
    return None


def _fallback_close_chain(instrument: str) -> Optional[float]:
    bq = _base_quote_codes(instrument)
    if bq is None:
        return None
    base, quote = bq
    c = _fxcm_d1_last_close(instrument)
    if c is not None:
        return c
    c = _frankfurter_mid(base, quote)
    if c is not None:
        return c
    return _open_er_api_mid(base, quote)


def _ny_calendar_today_naive() -> pd.Timestamp:
    """Calendar 'today' in America/New_York, midnight, timezone-naive (stable FX day for US users)."""
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def _bar_date_from_intraday_index(intraday: pd.DataFrame) -> pd.Timestamp:
    last_ts = pd.Timestamp(intraday.index[-1])
    try:
        last_utc = last_ts.tz_convert("UTC") if last_ts.tzinfo is not None else last_ts.tz_localize("UTC")
        return last_utc.tz_localize(None).normalize()
    except Exception:
        return pd.Timestamp(last_ts).normalize()


def _aggregate_intraday(intraday: pd.DataFrame) -> Optional[pd.DataFrame]:
    needed = {"Open", "High", "Low", "Close"}
    if intraday is None or intraday.empty or not needed.issubset(intraday.columns):
        return None
    today_date = _bar_date_from_intraday_index(intraday)
    o = float(pd.to_numeric(intraday["Open"].iloc[0], errors="coerce"))
    h = float(pd.to_numeric(intraday["High"].max(), errors="coerce"))
    lo = float(pd.to_numeric(intraday["Low"].min(), errors="coerce"))
    c = float(pd.to_numeric(intraday["Close"].iloc[-1], errors="coerce"))
    v = (
        float(pd.to_numeric(intraday["Volume"], errors="coerce").fillna(0.0).sum())
        if "Volume" in intraday.columns
        else 0.0
    )
    if any(pd.isna(x) for x in (o, h, lo, c)):
        return None
    return pd.DataFrame(
        {"Open": [o], "High": [h], "Low": [lo], "Close": [c], "Volume": [v]},
        index=pd.DatetimeIndex([today_date]),
    )


def _try_intraday(ticker: str, *, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return None
    try:
        raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception as e:
        logger.debug("Yahoo %s %s %s failed: %s", ticker, period, interval, e)
        return None
    return _aggregate_intraday(raw)


def _daily_carry_forward_bar(ticker: str) -> Optional[pd.DataFrame]:
    """Use last available **daily** close; if that day is before NY 'today', synthetic OHLC=today with that close."""
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return None
    try:
        daily = yf.Ticker(ticker).history(period="60d", interval="1d", auto_adjust=False)
    except Exception as e:
        logger.warning("Yahoo daily fallback failed for %s: %s", ticker, e)
        return None
    if daily is None or daily.empty:
        return None
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(daily.columns):
        return None
    daily = daily.copy()
    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index, utc=True)).tz_localize(None).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    r = daily.iloc[-1]
    last_d = pd.Timestamp(daily.index[-1]).normalize()
    anchor = _ny_calendar_today_naive()
    c = float(pd.to_numeric(r["Close"], errors="coerce"))
    if pd.isna(c):
        return None

    if last_d > anchor:
        logger.warning("Daily last bar %s after NY today %s — using last close on today's anchor.", last_d, anchor)

    if last_d == anchor or last_d > anchor:
        o = float(pd.to_numeric(r["Open"], errors="coerce"))
        h = float(pd.to_numeric(r["High"], errors="coerce"))
        lo = float(pd.to_numeric(r["Low"], errors="coerce"))
        v = float(pd.to_numeric(r.get("Volume", 0.0), errors="coerce")) if "Volume" in daily.columns else 0.0
        if pd.isna(o):
            o = c
        if pd.isna(h):
            h = c
        if pd.isna(lo):
            lo = c
        return pd.DataFrame(
            {"Open": [o], "High": [h], "Low": [lo], "Close": [c], "Volume": [v]},
            index=pd.DatetimeIndex([anchor]),
        )

    # Stale daily (e.g. last bar Friday, today Monday): treat **last close as today's close**.
    logger.info(
        "Live quote: daily last bar %s < NY today %s — synthetic today bar at last close %.5f",
        last_d.date(),
        anchor.date(),
        c,
    )
    return pd.DataFrame(
        {"Open": [c], "High": [c], "Low": [c], "Close": [c], "Volume": [0.0]},
        index=pd.DatetimeIndex([anchor]),
    )


def fetch_live_today_bar(instrument: str) -> Optional[pd.DataFrame]:
    """One-row OHLCV for the current session / NY calendar today, or ``None``."""
    ticker = YAHOO_FX_TICKERS.get(instrument)
    if ticker is None:
        logger.debug("No Yahoo mapping for %s; skip live quote.", instrument)
        return None

    try:
        import yfinance as yf  # noqa: F401, PLC0415
    except ImportError:
        logger.warning("yfinance not installed; cannot inject live quote for %s", instrument)
        return None

    for period, interval in (
        ("7d", "1m"),
        ("7d", "5m"),
        ("14d", "15m"),
        ("14d", "1h"),
    ):
        out = _try_intraday(ticker, period=period, interval=interval)
        if out is not None:
            logger.info("Live quote for %s via Yahoo %s %s (index %s)", instrument, period, interval, out.index[0])
            return out

    logger.info("Yahoo intraday empty for %s — trying daily carry-forward.", instrument)
    daily = _daily_carry_forward_bar(ticker)
    if daily is not None:
        return daily

    anchor = _ny_calendar_today_naive()
    spot = _fallback_close_chain(instrument)
    if spot is not None:
        logger.warning(
            "Live quote for %s: Yahoo unavailable — using fallback spot %.5f on NY anchor %s",
            instrument,
            spot,
            anchor.date(),
        )
        return _synthetic_anchor_bar(anchor, spot)

    logger.error("All live-quote sources failed for %s", instrument)
    return None


def _hourly_slice_ny_today(h: pd.DataFrame, *, anchor_ny: pd.Timestamp) -> pd.DataFrame:
    """Rows whose **America/New_York** calendar date equals ``anchor_ny`` (date match)."""
    if h is None or h.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(pd.to_datetime(h.index))
    if idx.tz is None:
        idx = idx.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
    else:
        idx = idx.tz_convert("UTC")
    idx_ny = idx.tz_convert("America/New_York").normalize()
    want = anchor_ny.normalize()
    mask = idx_ny == want
    return h.loc[mask]


def fetch_hourly_today_aggregated_bar(instrument: str) -> Optional[pd.DataFrame]:
    """One-row OHLC for **NY calendar today** from Yahoo **1h** bars (session-so-far)."""
    ticker = YAHOO_FX_TICKERS.get(instrument)
    if ticker is None:
        return None
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return None
    try:
        h = yf.Ticker(ticker).history(period="10d", interval="1h", auto_adjust=False)
    except Exception as e:
        logger.debug("Yahoo 1h history failed for %s: %s", instrument, e)
        return None
    if h is None or h.empty:
        return None
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(h.columns):
        return None
    anchor = _ny_calendar_today_naive()
    anchor_ny = pd.Timestamp(anchor).tz_localize("America/New_York").normalize()
    day = _hourly_slice_ny_today(h, anchor_ny=anchor_ny)
    if day.empty:
        logger.info("No 1h rows for NY today for %s — hourly synthetic skipped.", instrument)
        return None
    o = float(pd.to_numeric(day["Open"].iloc[0], errors="coerce"))
    hi = float(pd.to_numeric(day["High"].max(), errors="coerce"))
    lo = float(pd.to_numeric(day["Low"].min(), errors="coerce"))
    c = float(pd.to_numeric(day["Close"].iloc[-1], errors="coerce"))
    v = (
        float(pd.to_numeric(day["Volume"], errors="coerce").fillna(0.0).sum())
        if "Volume" in day.columns
        else 0.0
    )
    if any(pd.isna(x) for x in (o, hi, lo, c)):
        return None
    logger.info(
        "Hourly synthetic today for %s: O=%.5f H=%.5f L=%.5f C=%.5f (%d bars)",
        instrument,
        o,
        hi,
        lo,
        c,
        len(day),
    )
    return pd.DataFrame(
        {"Open": [o], "High": [hi], "Low": [lo], "Close": [c], "Volume": [v]},
        index=pd.DatetimeIndex([anchor]),
    )


def append_hourly_today_bar(raw: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Prefer **1h NY-today** OHLC; fall back to :func:`append_live_today_bar`."""
    if raw is None or raw.empty:
        return raw
    new_row = fetch_hourly_today_aggregated_bar(instrument)
    if new_row is None:
        return append_live_today_bar(raw, instrument)
    today = pd.Timestamp(new_row.index[0]).normalize()
    out = raw.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if today in out.index:
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in out.columns and col in new_row.columns:
                out.loc[today, col] = float(new_row[col].iloc[0])
        return out.sort_index()
    return pd.concat([out, new_row], axis=0).sort_index()


def append_live_today_bar(raw: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Append or refresh the **anchor** (NY calendar today) row from Yahoo / daily fallback."""
    if raw is None or raw.empty:
        return raw
    new_row = fetch_live_today_bar(instrument)
    if new_row is None:
        return raw

    today = pd.Timestamp(new_row.index[0]).normalize()
    out = raw.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if today in out.index:
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in out.columns and col in new_row.columns:
                out.loc[today, col] = float(new_row[col].iloc[0])
        return out.sort_index()

    return pd.concat([out, new_row], axis=0).sort_index()
