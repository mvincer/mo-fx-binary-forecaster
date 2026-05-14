"""
Load **fundamentals_daily** + **technicals_daily** + raw FX parquets from the ETF Forecaster
``fx_data_collect`` repository (fresh Yahoo/FRED-backed panel).

Set ``FXNL_DATA_REPO`` to the ``.../fx_data_collect/data`` folder, or rely on the default
sibling layout: ``../ETF Forecaster/fx_data_collect/data``.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ``inherit`` = use ``FXNL_INJECT_LIVE_QUOTE`` + ``append_live_today_bar`` (default).
# ``off`` = parquet only (no synthetic today bar).
_REPO_FX_INJECT: ContextVar[str] = ContextVar("_REPO_FX_INJECT", default="inherit")

THIS_ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def repo_fx_inject_context(mode: str) -> Iterator[None]:
    """Temporarily set repo raw-FX live row behaviour for ``assemble_merged_from_repo`` / OHLC loads.

    ``mode``: ``inherit`` | ``off``
    """
    if mode not in ("inherit", "off"):
        raise ValueError(f"unknown repo_fx inject mode: {mode}")
    tok: Token = _REPO_FX_INJECT.set(mode)
    try:
        yield
    finally:
        _REPO_FX_INJECT.reset(tok)


def _repo_fx_inject_mode() -> str:
    return _REPO_FX_INJECT.get()


def _load_dotenv_paths() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(THIS_ROOT / ".env")
        load_dotenv(THIS_ROOT.parent / "Currencies" / ".env")
        load_dotenv(THIS_ROOT.parent / "ETF Forecaster" / ".env")
    except Exception:
        pass


def data_repo_root() -> Path | None:
    """Return path to ``fx_data_collect/data`` if present and complete, else ``None``."""
    _load_dotenv_paths()
    env = (os.environ.get("FXNL_DATA_REPO") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if _repo_ok(p):
            return p
        logger.warning("FXNL_DATA_REPO set but invalid or missing model_inputs: %s", p)

    candidates = [
        THIS_ROOT.parent / "ETF Forecaster" / "fx_data_collect" / "data",
        THIS_ROOT.parent.parent / "ETF Forecaster" / "fx_data_collect" / "data",
    ]
    for c in candidates:
        if _repo_ok(c):
            return c.resolve()
    return None


def _repo_ok(p: Path) -> bool:
    return (
        p.is_dir()
        and (p / "model_inputs" / "fundamentals_daily.parquet").is_file()
        and (p / "model_inputs" / "technicals_daily.parquet").is_file()
        and (p / "raw" / "fx").is_dir()
    )


def use_repo_data() -> bool:
    """True when env requests repo and files exist."""
    _load_dotenv_paths()
    v = (os.environ.get("FXNL_USE_REPO_DATA") or "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return data_repo_root() is not None


def pair_id_from_instrument(instrument: str) -> str:
    return instrument.replace("/", "_").replace(" ", "_")


def trim_period(df: pd.DataFrame, period: str | None) -> pd.DataFrame:
    if df.empty:
        return df
    if not period or str(period).lower() in ("max", "all", "", "none"):
        return df.sort_index()
    s = str(period).strip().lower()
    end = pd.Timestamp(df.index.max())
    if s.endswith("d") and s[:-1].isdigit():
        start = end - pd.Timedelta(days=int(s[:-1]))
        return df.loc[df.index >= start].sort_index()
    if s.endswith("y") and s[:-1].isdigit():
        start = end - pd.Timedelta(days=365 * int(s[:-1]))
        return df.loc[df.index >= start].sort_index()
    return df.sort_index()


def _strip_tz_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).tz_localize(None).normalize()
    return out.sort_index()


def _live_quote_enabled() -> bool:
    """Truthy ``FXNL_INJECT_LIVE_QUOTE`` env var enables today-bar synthesis."""
    v = (os.environ.get("FXNL_INJECT_LIVE_QUOTE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _read_raw_fx_with_optional_live(fx_path: Path, instrument: str) -> pd.DataFrame:
    """Read ``fx_path`` parquet and optionally append / refresh today's row."""
    raw = pd.read_parquet(fx_path)
    mode = _repo_fx_inject_mode()
    if mode == "off":
        return raw
    # inherit
    if _live_quote_enabled():
        try:
            from fxnl.live_quote import append_live_today_bar  # noqa: PLC0415

            raw = append_live_today_bar(raw, instrument)
        except Exception as e:
            logger.warning("Live-quote injection failed for %s: %s", instrument, e)
    return raw


def assemble_merged_from_repo(
    instrument: str,
    *,
    period: str | None,
    study_bars: int | None,
) -> tuple[pd.DataFrame, str, str]:
    """
    Mirror :func:`src.merged_data.assemble_merged_panel` using repo parquets.

    Returns ``(merged, pid, close_col_name)``.
    """
    root = data_repo_root()
    if root is None:
        return pd.DataFrame(), "", ""

    pid = pair_id_from_instrument(instrument)
    close_col = f"{pid}_close"
    fx_path = root / "raw" / "fx" / f"{pid}.parquet"
    if not fx_path.is_file():
        logger.warning("Repo missing FX parquet: %s", fx_path)
        return pd.DataFrame(), pid, close_col

    raw = _read_raw_fx_with_optional_live(fx_path, instrument)
    raw = _strip_tz_index(raw)
    if raw.empty or "Close" not in raw.columns:
        return pd.DataFrame(), pid, close_col
    raw = trim_period(raw, period)

    try:
        fund = pd.read_parquet(root / "model_inputs" / "fundamentals_daily.parquet")
        tech = pd.read_parquet(root / "model_inputs" / "technicals_daily.parquet")
    except Exception as e:
        logger.error("Failed reading model_inputs: %s", e)
        return pd.DataFrame(), pid, close_col

    fund = _strip_tz_index(fund) if len(fund) else fund
    tech = _strip_tz_index(tech) if len(tech) else tech

    ix = raw.index
    fund_a = fund.reindex(ix).ffill()
    pref = f"{pid}_"
    tech_cols = [c for c in tech.columns if str(c).startswith(pref)]
    tech_p = tech[tech_cols].reindex(ix).ffill() if tech_cols else pd.DataFrame(index=ix)

    close_s = raw["Close"].astype(float).rename(close_col)
    parts = [fund_a, tech_p, close_s.to_frame()]
    merged = pd.concat(parts, axis=1)
    merged = merged.loc[raw.index]
    merged = merged.dropna(axis=1, how="all")
    merged = merged.dropna(subset=[close_col])

    fc = [c for c in merged.columns if str(c).startswith("fund_")]
    if fc:
        merged[fc] = merged[fc].ffill()

    n_cols = merged.shape[1]
    thresh_non_na = max(min(int(n_cols * 0.25), n_cols), 5)
    merged = merged.dropna(thresh=thresh_non_na)

    if study_bars is not None and int(study_bars) > 0:
        merged = merged.tail(int(study_bars))

    return merged, pid, close_col


def load_primary_fx_ohlc(
    primary_instrument: str,
    period: str | None,
    study_bars: int | None,
    index: pd.Index,
) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """High / Low / Close aligned to ``index`` from repo raw FX parquet."""
    root = data_repo_root()
    if root is None:
        return None
    pid = pair_id_from_instrument(primary_instrument)
    p = root / "raw" / "fx" / f"{pid}.parquet"
    if not p.is_file():
        return None
    raw = _read_raw_fx_with_optional_live(p, primary_instrument)
    raw = _strip_tz_index(raw)
    if raw.empty or not all(c in raw.columns for c in ("High", "Low", "Close")):
        return None
    raw = trim_period(raw, period)
    if study_bars is not None and int(study_bars) > 0:
        raw = raw.tail(int(study_bars))
    hi = raw["High"].astype(float).reindex(index)
    lo = raw["Low"].astype(float).reindex(index)
    cl = raw["Close"].astype(float).reindex(index)
    return hi, lo, cl


def repo_manifest_dates() -> dict[str, Any]:
    """Best-effort last-bar info for UI (replaces stale Currencies parquet caption)."""
    root = data_repo_root()
    if root is None:
        return {}
    out: dict[str, Any] = {"repo_root": str(root)}
    mj = root / "model_inputs" / "manifest.json"
    if mj.is_file():
        try:
            out["model_inputs_manifest"] = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            pass
    raw_fx = root / "raw" / "fx"
    if raw_fx.is_dir():
        ends = []
        for pq in sorted(raw_fx.glob("*.parquet")):
            try:
                df = pd.read_parquet(pq)
                if len(df.index):
                    ends.append(pd.Timestamp(df.index.max()))
            except Exception:
                continue
        if ends:
            out["fx_parquet_max_date"] = str(max(ends).date())
    return out

