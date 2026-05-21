"""FX daily session calendar — single source of truth for forecast dates.

Rule (5:00 PM **America/New_York** close):

- **Workday before 5 PM** → data ends at **previous business day's** close;
  ``target_forecast_date`` = **today** (the next session not yet closed).
- **Workday at/after 5 PM** → data ends at **today's** close;
  ``target_forecast_date`` = **next business day**.
- **Weekend** (Sat/Sun) → data ends at the **previous Friday's** close;
  ``target_forecast_date`` = **next Monday** (next business day after that Friday).

All returned timestamps are **timezone-naive, normalized to midnight**.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

NY_TZ = "America/New_York"
NY_CLOSE_HOUR = 17  # 5:00 PM


def _now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ)


def _is_workday(ts: pd.Timestamp) -> bool:
    return ts.weekday() < 5


def _after_close(ts: pd.Timestamp) -> bool:
    return ts.hour >= NY_CLOSE_HOUR


def fx_session_dates(
    now_ny: Optional[pd.Timestamp] = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(data_through_date, target_forecast_date)`` per the 5 PM NY rule."""
    now = now_ny if (now_ny is not None and getattr(now_ny, "tzinfo", None) is not None) else _now_ny()
    today_naive = now.normalize().tz_localize(None)
    bday = pd.tseries.offsets.BusinessDay()

    if _is_workday(now) and _after_close(now):
        data_through = today_naive
        target_forecast = (today_naive + bday).normalize()
    else:
        prev_bday = (today_naive - bday).normalize()
        data_through = prev_bday
        target_forecast = (prev_bday + bday).normalize()

    return data_through, target_forecast


def fx_data_through_date(now_ny: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    return fx_session_dates(now_ny)[0]


def fx_target_forecast_date(now_ny: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    return fx_session_dates(now_ny)[1]


def trim_index_to_data_through(
    index: pd.Index, now_ny: Optional[pd.Timestamp] = None
) -> pd.Index:
    """Drop any panel rows whose date is **after** ``fx_data_through_date``."""
    through = fx_data_through_date(now_ny)
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    return index[idx <= through]


def should_inject_live_today_bar(now_ny: Optional[pd.Timestamp] = None) -> bool:
    """Live today-bar injection is only valid **after** the NY 5 PM close.

    ``FXNL_INJECT_LIVE_QUOTE=force`` overrides for debugging; ``0/false/no/off`` disables.
    Any other truthy value defers to the 5 PM rule.
    """
    raw = (os.environ.get("FXNL_INJECT_LIVE_QUOTE") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw == "force":
        return True
    now = now_ny if (now_ny is not None and getattr(now_ny, "tzinfo", None) is not None) else _now_ny()
    return _is_workday(now) and _after_close(now)


def align_target_forecast_date(
    latest_feature_date: pd.Timestamp,
    *,
    align_to_next_bar: bool,
    lead: int = 1,
    now_ny: Optional[pd.Timestamp] = None,
) -> pd.Timestamp:
    """Resolve ``target_forecast_date`` for one model row.

    With ``align_to_next_bar=True`` (every lead targets the same session): the date is
    derived **solely** from the current clock via :func:`fx_target_forecast_date`, so the
    workbook never drifts off the 5 PM NY rule even if the panel index is stale.
    """
    if align_to_next_bar:
        return fx_target_forecast_date(now_ny)
    latest_norm = pd.Timestamp(latest_feature_date).normalize()
    return (latest_norm + pd.tseries.offsets.BusinessDay(int(lead))).normalize()
