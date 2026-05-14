"""
FX Non-Linear Forecast for Direction

Run: py -m streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
from dotenv import load_dotenv

# Load before fxnl.paths so FXNL_CURRENCIES_ROOT in .env applies to cache resolution.
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT.parent / "Currencies" / ".env")

_rs = str(_ROOT)
if _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from fxnl.paths import currencies_root, fxcm_primary_parquet_mtime, patch_currencies_sys_path

patch_currencies_sys_path()

import pandas as pd
import streamlit as st

from fxnl.backtest import (
    equity_curve,
    holdout_oos_parts,
    performance_stats,
    stitched_oos_frame,
    stitched_oos_summary,
    strategy_returns_binary,
    strategy_returns_ternary,
)
from fxnl.data_panel import build_direction_dataset, prepare_direction_frame
from fxnl.model_fit_core import DIRECTION_MODEL_OPTIONS
from fxnl.models_runner import run_direction_models_with_preds
from fxnl.rolling_tune_oos import rolling_tune_sequential_oos
from fxnl.walk_forward import walk_forward_evaluate


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


@st.cache_data(ttl=180, show_spinner="Building panel & fitting models…")
def _cached_holdout_run(
    primary: str,
    target_mode: str,
    lead: int,
    period_s: str,
    study_bars: int,
    context_key: str,
    pca_components: int,
    include_garch: bool,
    target_return_basis: str,
    models_filter: str,
) -> tuple[dict, pd.DataFrame, dict]:
    ctx = json.loads(context_key)
    sb = int(study_bars) if study_bars and study_bars > 0 else None
    per = period_s.strip() or None
    dset = build_direction_dataset(
        primary,
        context_instruments=ctx,
        period=per,
        study_bars=sb,
        lead=lead,
        target_mode=target_mode,
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=include_garch,
        target_return_basis=target_return_basis,
    )
    if dset.get("error"):
        return dset, pd.DataFrame(), {}
    pca_k = int(pca_components) if pca_components and pca_components > 0 else None
    mf = str(models_filter).strip()
    m_inc = None if mf.upper() == "ALL" else [mf]
    res, preds = run_direction_models_with_preds(
        dset,
        random_state=42,
        global_pca_components=pca_k,
        skip_lstm=False,
        models_include=m_inc,
    )
    return dset, res, preds


@st.cache_data(ttl=300, show_spinner="Walk-forward evaluation (can take several minutes)…")
def _cached_walk_forward(
    primary: str,
    target_mode: str,
    lead: int,
    period_s: str,
    study_bars: int,
    context_key: str,
    pca_components: int,
    include_garch: bool,
    lookback_bars: int,
    window_size: int,
    train_frac: float,
    step: int,
    skip_lstm: bool,
    target_return_basis: str,
    models_filter: str,
) -> dict:
    ctx = json.loads(context_key)
    sb = int(study_bars) if study_bars and study_bars > 0 else None
    per = period_s.strip() or None
    base = prepare_direction_frame(
        primary,
        context_instruments=ctx,
        period=per,
        study_bars=sb,
        lead=lead,
        target_mode=target_mode,
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=include_garch,
        target_return_basis=target_return_basis,
    )
    if base.get("error"):
        return {"error": base["error"], "base": base}
    n_full = len(base["X"])
    ix_full = base["X"].index
    min_full = ix_full.min()
    max_full = ix_full.max()
    lb = int(lookback_bars) if lookback_bars and int(lookback_bars) > 0 else 0
    trimmed = lb > 0 and n_full > lb
    panel_before_lookback = {
        "n_rows_full": int(n_full),
        "min_date_full": min_full,
        "max_date_full": max_full,
        "lookback_bars": lb,
        "trimmed": trimmed,
    }
    if trimmed:
        base = dict(base)
        base["X"] = base["X"].iloc[-lb:]
        base["y"] = base["y"].iloc[-lb:]
        base["forward_r"] = base["forward_r"].iloc[-lb:]
        base["dates"] = pd.DatetimeIndex(pd.to_datetime(base["dates"]))[-lb:]
        if base.get("hi_exc") is not None:
            base["hi_exc"] = base["hi_exc"].iloc[-lb:]
        if base.get("lo_exc") is not None:
            base["lo_exc"] = base["lo_exc"].iloc[-lb:]
        base["n_total"] = len(base["X"])
    pca_k = int(pca_components) if pca_components and pca_components > 0 else None
    mf = str(models_filter).strip()
    m_inc = None if mf.upper() == "ALL" else [mf]
    wf = walk_forward_evaluate(
        base["X"],
        base["y"],
        base["forward_r"],
        window_size=int(window_size),
        train_frac=float(train_frac),
        step=int(step) if int(step) > 0 else None,
        lookback_bars=int(lookback_bars) if int(lookback_bars) > 0 else None,
        random_state=42,
        global_pca_components=pca_k,
        skip_lstm=bool(skip_lstm),
        pair=str(base.get("primary_id", "")),
        lead=int(lead),
        target_mode=target_mode,
        models_include=m_inc,
    )
    return {"base": base, "wf": wf, "panel_before_lookback": panel_before_lookback}


@st.cache_data(ttl=600, show_spinner="Rolling tune + sequential 1-step OOS (can be very slow)…")
def _cached_rolling_tune(
    primary: str,
    target_mode: str,
    lead: int,
    period_s: str,
    study_bars: int,
    context_key: str,
    include_garch: bool,
    lookback_bars: int,
    history_pool: int,
    fit_window: int,
    train_frac_inner: float,
    forecast_step: int,
    max_steps_rt: int,
    family_rt: str,
    target_return_basis: str,
    expanding_pool: bool,
    inner_test_frac: float,
    sticky_champion: bool,
    rolling_metrics_chunk: int,
    tune_n_jobs: int,
    rolling_pca_components: int,
) -> dict:
    """Build the same panel as walk-forward, then run per-bar tuned models (no global PCA in this path)."""
    ctx = json.loads(context_key)
    sb = int(study_bars) if study_bars and study_bars > 0 else None
    per = period_s.strip() or None
    base = prepare_direction_frame(
        primary,
        context_instruments=ctx,
        period=per,
        study_bars=sb,
        lead=lead,
        target_mode=target_mode,
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=include_garch,
        target_return_basis=target_return_basis,
    )
    if base.get("error"):
        return {"error": base["error"], "base": base}
    n_full = len(base["X"])
    ix_full = base["X"].index
    min_full = ix_full.min()
    max_full = ix_full.max()
    lb = int(lookback_bars) if lookback_bars and int(lookback_bars) > 0 else 0
    trimmed = lb > 0 and n_full > lb
    panel_before_lookback = {
        "n_rows_full": int(n_full),
        "min_date_full": min_full,
        "max_date_full": max_full,
        "lookback_bars": lb,
        "trimmed": trimmed,
    }
    if trimmed:
        base = dict(base)
        base["X"] = base["X"].iloc[-lb:]
        base["y"] = base["y"].iloc[-lb:]
        base["forward_r"] = base["forward_r"].iloc[-lb:]
        base["dates"] = pd.DatetimeIndex(pd.to_datetime(base["dates"]))[-lb:]
        if base.get("hi_exc") is not None:
            base["hi_exc"] = base["hi_exc"].iloc[-lb:]
        if base.get("lo_exc") is not None:
            base["lo_exc"] = base["lo_exc"].iloc[-lb:]
        base["n_total"] = len(base["X"])

    ms = None if int(max_steps_rt) <= 0 else int(max_steps_rt)
    rt = rolling_tune_sequential_oos(
        base["X"],
        base["y"],
        base["forward_r"],
        history_bars=int(history_pool),
        fit_window_bars=int(fit_window),
        train_frac=float(train_frac_inner),
        step=int(forecast_step),
        family_filter=str(family_rt).strip().lower(),
        max_steps=ms,
        random_state=42,
        expanding_pool=bool(expanding_pool),
        history_cap_bars=None,
        inner_test_frac=float(inner_test_frac),
        sticky_champion=bool(sticky_champion),
        rolling_metrics_chunk=int(rolling_metrics_chunk),
        tune_n_jobs=int(tune_n_jobs),
        global_pca_components=int(rolling_pca_components)
        if rolling_pca_components and int(rolling_pca_components) > 0
        else None,
    )
    return {"base": base, "rt": rt, "panel_before_lookback": panel_before_lookback}


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fxcm_meta(primary: str, parquet_mtime: float) -> dict:
    """FXCM parquet stats; ``parquet_mtime`` busts cache when the file on disk changes after refresh."""
    _ = parquet_mtime  # Streamlit cache key only — must change when parquet file is replaced
    patch_currencies_sys_path()
    try:
        from src.fxcm_loader import instrument_cache_meta  # noqa: PLC0415

        return instrument_cache_meta(primary)
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_feature_data_bounds(
    primary: str,
    target_mode: str,
    lead: int,
    period_s: str,
    study_bars: int,
    context_key: str,
    include_garch: bool,
    target_return_basis: str,
    parquet_mtime: float,
) -> dict[str, Any]:
    """
    Same pipeline as **Run evaluation**: cleaned feature matrix + labels after merges and NaN filters.
    Used only for calendar min/max (cheap vs full model fits thanks to cache).
    ``parquet_mtime`` ties invalidation to the primary pair FXCM file (same as meta banner).
    """
    _ = parquet_mtime  # Streamlit cache key only
    ctx = json.loads(context_key)
    sb = int(study_bars) if study_bars and study_bars > 0 else None
    per = period_s.strip() or None
    base = prepare_direction_frame(
        primary,
        context_instruments=ctx,
        period=per,
        study_bars=sb,
        lead=lead,
        target_mode=target_mode,
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=include_garch,
        target_return_basis=target_return_basis,
    )
    warn = None
    err_key = base.get("error")
    if err_key and "X" in base and base["X"] is not None and len(base["X"]) > 0:
        warn = str(err_key)
    elif err_key:
        return {"ok": False, "error": str(err_key)}
    if "X" not in base or base["X"] is None or len(base["X"]) == 0:
        return {"ok": False, "error": str(err_key or "empty features")}
    ix = base["X"].index
    dti = pd.DatetimeIndex(pd.to_datetime(ix))
    out: dict[str, Any] = {
        "ok": True,
        "feature_min": dti.min(),
        "feature_max": dti.max(),
        "n_rows": int(len(base["X"])),
        "primary_id": str(base.get("primary_id", "")),
    }
    if warn:
        out["warning"] = warn
    return out


def _show_fxcm_cache_status(primary: str) -> None:
    """Explain that feature dates are capped by on-disk history; nudge refresh if cache is old."""
    try:
        from fxnl.data_repo import data_repo_root, repo_manifest_dates, use_repo_data

        if use_repo_data():
            root = data_repo_root()
            meta = repo_manifest_dates()
            mx = meta.get("fx_parquet_max_date") or "?"
            st.success(
                f"**Using fresh ETF data repo** (`fx_data_collect`): `{root}`. "
                f"Latest FX parquet end date seen: **{mx}**. "
                "Rebuild panels with `py -m fx_data_collect.run_model_inputs` after updating raw FX."
            )
            return
    except Exception:
        pass

    mt = fxcm_primary_parquet_mtime(primary)
    meta = _cached_fxcm_meta(primary, mt)
    if not meta:
        st.caption(
            f"No FXCM parquet found for **{primary}**. Use the **Currencies** app → "
            "**Download / refresh market data now**, then return here."
        )
        return
    end_s = meta.get("end")
    start_s = meta.get("start")
    rows = meta.get("rows", 0)
    mtime = meta.get("last_modified_utc", "")
    path_disp = meta.get("path") or ""
    st.caption(
        f"**Where dates come from:** features use this **on-disk** parquet for **{primary}** (not live quotes): "
        f"`{path_disp}`  ·  Index span **{start_s}** → **{end_s}** ({rows:,} rows; file mtime UTC **{mtime}**). "
        "**`period` = max** cannot extend past the **last bar** in that file. "
        "If you refreshed elsewhere but still see an old end date, confirm this path is the **same** folder the "
        "Currencies app writes to (duplicate projects / OneDrive copies are a common mismatch)."
    )
    if not end_s:
        return
    end = pd.to_datetime(end_s).normalize()
    today = pd.Timestamp.now().normalize()
    age_days = int((today - end).days)
    if age_days > 14:
        st.warning(
            f"The **last bar in that parquet** is **{end.date()}** (~**{age_days}** calendar days ago). "
            "Refresh in **Currencies** → **Download / refresh market data now**, then **rerun this page** (or press **R**). "
            "If the date stays old, open the path above in Explorer and check its **modified time**—your refresh may "
            "have saved to a **different** Currencies clone than this app resolves. "
            f"This app reads from: `{currencies_root()}`. "
            "To point at another checkout, set **`FXNL_CURRENCIES_ROOT`** in **`.env`** (absolute path to that Currencies folder) "
            "and restart Streamlit."
        )


def _panel_calendar_range(d: dict | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not d or d.get("error") or "X" not in d:
        return None, None
    ix = d["X"].index
    if not isinstance(ix, pd.DatetimeIndex):
        ix = pd.DatetimeIndex(pd.to_datetime(ix))
    if len(ix) == 0:
        return None, None
    return ix.min(), ix.max()


def _data_preview_table(d: dict) -> pd.DataFrame | None:
    if d.get("error") or "X" not in d:
        return None
    X = d["X"]
    y = d.get("y")
    r = d.get("forward_r")
    out = X.copy()
    out.insert(0, "date", pd.to_datetime(out.index))
    if y is not None and len(y) == len(out):
        out["y"] = y.reindex(out.index).values
    if r is not None and len(r.reindex(out.index)) == len(out):
        out["forward_r"] = r.reindex(out.index).values
    for k in ("hi_exc", "lo_exc"):
        s = d.get(k)
        if s is not None and len(s.reindex(out.index)) == len(out):
            out[k] = s.reindex(out.index).values
    out = out.sort_index()
    return out


def _run_backtest_charts(
    oos_parts: list,
    model_names: list[str],
    target_mode: str,
    trade_mode: str,
    prob_threshold: float,
    oos_tail: int = 0,
    *,
    sequential_oos: bool = False,
) -> None:
    summ = stitched_oos_summary(oos_parts)
    if sequential_oos:
        st.caption(
            "**Sequential 1-step OOS (rolling tune):** stitched **one prediction per origin** in time order "
            f"(each bar refit with tuned hyperparameters). Calendar span: **{summ['min_date']}** → **{summ['max_date']}** "
            f"({summ.get('n_unique_dates', summ['n_rows'])} dates). Equity uses **forward_r** at each prediction date."
        )
    else:
        st.caption(
            "**Pooled OOS backtest (walk-forward):** all out-of-sample **test** rows from every successful fold, "
            f"stitched in time order. Calendar span of those test dates: **{summ['min_date']}** → **{summ['max_date']}** "
            f"({summ.get('n_unique_dates', summ['n_rows'])} unique dates; {summ['n_rows']} row-slots across folds). "
            "This is **not** “only the latest window”—the curve usually starts after the first full train window."
        )
    if oos_tail and oos_tail > 0:
        st.caption(f"Charts below use the **last {int(oos_tail)}** OOS rows per model (after dedupe), not the full span.")
    tm = "threshold" if "Mode 2" in trade_mode else "always"
    for m in model_names:
        f = stitched_oos_frame(oos_parts, m)
        if f is None or f.empty:
            st.warning(f"No OOS data for {m}.")
            continue
        if oos_tail and oos_tail > 0:
            f = f.sort_values("date").tail(int(oos_tail)).reset_index(drop=True)
        st.markdown(f"**{m}**")
        st.caption(f"Plotted segment: **{f['date'].min()}** → **{f['date'].max()}** · {len(f)} rows.")
        if target_mode == "binary":
            sr, pos = strategy_returns_binary(f, mode=tm, prob_threshold=prob_threshold)
        else:
            sr, pos = strategy_returns_ternary(f, mode=tm, prob_threshold=prob_threshold)
        stats = performance_stats(sr, positions=pos)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Ann. return", f"{stats['ann_return']*100:.2f}%")
        c2.metric("Sharpe", f"{stats['sharpe']:.2f}")
        c3.metric("Sortino", f"{stats['sortino']:.2f}")
        c4.metric("Max DD", f"{stats['max_drawdown']*100:.2f}%")
        c5.metric("Hit rate (trades)", f"{stats['hit_rate']*100:.1f}%")
        c6.metric("% days traded", f"{stats['pct_traded']*100:.1f}%")
        eq = equity_curve(sr, dates=f["date"])
        st.line_chart(eq.set_index("date")["equity"], height=220)


def main() -> None:
    st.set_page_config(page_title="FX Non-Linear Forecast for Direction", layout="wide")
    st.title("FX Non-Linear Forecast for Direction")
    st.caption(
        "**Hold-out** = one 50/50 time split. **Walk-forward** = rolling window with train/test per fold. "
        "**Rolling tune** = each origin uses a **history pool** + inner train/val to pick hyperparameters, "
        "then refits on the pool and predicts **one step**; repeats bar-by-bar. "
        "GARCH vol features: `feat_garch_*` (see `fxnl/vol_features.py`)."
    )

    try:
        from src.config import FX_PAIR_INSTRUMENTS  # type: ignore  # noqa: PLC0415
        from src.cache_paths import FXCM_DIR  # type: ignore  # noqa: PLC0415
    except Exception as e:
        st.error(f"Could not import Currencies config: {e}")
        try:
            st.info(f"Expected Currencies at: `{currencies_root()}`")
        except Exception:
            pass
        return

    if not FXCM_DIR.exists() or not any(FXCM_DIR.glob("*.parquet")):
        st.warning(
            f"No FXCM parquet in `{FXCM_DIR}`. Download history from the **Currencies** app "
            "(**Download / refresh market data now**) first."
        )

    eval_mode = st.radio(
        "Evaluation mode",
        [
            "Walk-forward 80/20 (rolling)",
            "Rolling tune → sequential 1-step OOS",
            "Hold-out 50/50 (single split)",
        ],
        horizontal=True,
    )
    mode_wf = eval_mode.startswith("Walk-forward")
    mode_rt = "Rolling tune" in eval_mode
    mode_ho = eval_mode.startswith("Hold-out")
    default_period = "max"
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        primary = st.selectbox("Primary pair (target)", FX_PAIR_INSTRUMENTS, index=0)
    with col_b:
        target_mode = st.radio(
            "Target",
            ["binary", "ternary"],
            horizontal=True,
            help="Binary: up vs down. Ternary: <−0.5%, mid, >+0.5% (forward simple return).",
        )
    with col_c:
        lead = st.slider("Lead (days ahead)", min_value=1, max_value=10, value=1)

    _show_fxcm_cache_status(primary)

    period = st.text_input("History window (`period`)", value=default_period, help="e.g. max, 730d, 5y")
    study_bars = st.number_input(
        "Latest N daily bars (0 = all in window)",
        min_value=0,
        value=0,
        step=250,
    )
    pca_components = st.number_input(
        "Global PCA components (0 = use full feature matrix)",
        min_value=0,
        value=0,
        step=1,
    )
    include_garch = st.checkbox("Include GARCH / vol features (rolling GARCH + realized vol)", value=True)
    target_basis_choice = st.radio(
        "Target label (what we predict)",
        ["close", "high_low"],
        format_func=lambda x: (
            "Close-to-close net change over lead"
            if x == "close"
            else "High/Low path vs entry close (max High, min Low over next lead bars)"
        ),
        horizontal=True,
        help="High/Low mode labels direction only if intraday extremes support it; forward_r for backtest stays close-to-close.",
    )
    others = [p for p in FX_PAIR_INSTRUMENTS if p != primary]
    context = st.multiselect(
        "Other pairs’ technical features to merge",
        others,
        default=others,
    )
    context_full = [primary, *sorted(set(context))]
    ctx_key = json.dumps(context_full, sort_keys=True)

    st.subheader("Data range check (current sidebar settings)")
    c_fx, c_fe = st.columns(2)
    _fx_mt = fxcm_primary_parquet_mtime(primary)
    meta_ck = _cached_fxcm_meta(primary, _fx_mt)
    bounds = _cached_feature_data_bounds(
        primary,
        target_mode,
        int(lead),
        period,
        int(study_bars),
        ctx_key,
        include_garch,
        target_basis_choice,
        _fx_mt,
    )
    with c_fx:
        st.markdown("**Primary pair — FXCM cache on disk**")
        if meta_ck.get("end"):
            st.metric(
                "Last daily bar",
                str(pd.to_datetime(meta_ck["end"]).date()),
                help="From the Currencies project FXCM parquet for the selected primary pair.",
            )
            _nr = meta_ck.get("rows")
            _nr_s = f"{int(_nr):,}" if isinstance(_nr, int) else "—"
            st.caption(
                f"First bar: **{meta_ck.get('start', '—')}**  ·  rows: **{_nr_s}**  ·  "
                f"file mtime (UTC): **{meta_ck.get('last_modified_utc', '—')}**"
            )
        else:
            st.warning("No FXCM parquet found for this pair. Refresh in the Currencies app.")
    with c_fe:
        st.markdown("**Feature & label panel** (merged + cleaned, before training)")
        if bounds.get("ok"):
            st.metric(
                "Last row date",
                str(pd.to_datetime(bounds["feature_max"]).date()),
            )
            st.caption(
                f"First row: **{pd.to_datetime(bounds['feature_min']).date()}**  ·  "
                f"**{bounds['n_rows']:,}** rows  ·  primary id: `{bounds.get('primary_id', '')}`"
            )
            if bounds.get("warning"):
                st.warning(f"Panel build note: {bounds['warning']}")
        else:
            st.error(f"Could not build panel: **{bounds.get('error', 'unknown')}**")
    if meta_ck.get("end") and bounds.get("ok"):
        fe = pd.to_datetime(bounds["feature_max"]).normalize()
        fx = pd.to_datetime(meta_ck["end"]).normalize()
        if fx > fe + pd.Timedelta(days=7):
            st.info(
                "The **raw FX cache** runs later than the **feature panel**. Trailing days are often dropped by "
                "FRED/merge NaN rules, GARCH warm-up, `study_bars`, or **lead/label** alignment (no `y` on the very last rows). "
                "That is expected unless you need features through the same calendar date as the raw file."
            )

    lookback_bars = 0
    window_size = 252
    train_frac = 0.8
    step = 0
    skip_lstm_wf = True
    rt_history_pool = 1000
    rt_fit_window = 250
    rt_train_frac_inner = 0.8
    rt_forecast_step = 1
    rt_max_steps = 250
    rt_family = "fast"
    rt_expanding_pool = True
    rt_inner_test_frac = 0.15
    rt_sticky_champion = False
    rt_rolling_chunk = 100
    rt_tune_n_jobs = -1
    rt_rolling_pca = 5

    if mode_wf:
        st.subheader("Walk-forward parameters")
        w1, w2, w3, w4, w5 = st.columns(5)
        with w1:
            lookback_bars = st.number_input(
                "History length (bars, 0 = all)",
                min_value=0,
                value=0,
                step=100,
                help=(
                    "Walk-forward uses only the **last N** rows of the cleaned feature matrix. "
                    "**Default 0** = use the full cleaned panel. "
                    "Setting e.g. 1000 **drops older history** and caps rows at 1000 — "
                    "your preview will never show years before that window unless you lower N or use 0."
                ),
                key="wf_lookback_bars",
            )
        with w2:
            window_size = st.number_input(
                "Window size (bars per fold)", min_value=80, value=252, step=20, key="wf_window_size"
            )
        with w3:
            train_frac = st.slider(
                "Train fraction within window",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05,
                key="wf_train_frac",
            )
        with w4:
            step = st.number_input(
                "Step (0 = test-length)",
                min_value=0,
                value=0,
                step=5,
                help="Slide window by this many rows; 0 → step = round(window × (1 − train fraction)).",
                key="wf_step",
            )
        with w5:
            skip_lstm_wf = st.checkbox("Skip LSTM in walk-forward (faster)", value=True, key="wf_skip_lstm")

    elif mode_rt:
        st.subheader("Rolling tune + sequential OOS")
        st.caption(
            "Uses **history pool** rows ending before each forecast origin; **inner window** for train/val/(test) tuning; "
            "chosen variant refits on the **pool**, then predicts **one bar**. Enable **expanding pool** so a **1000-bar** panel "
            "can produce ~750 origins with a **250** inner window (first origin at *t* = 250). "
            "**Inner test %** picks one winner across families by held-out test accuracy; **sticky champion** can keep the prior winner "
            "when its test score still beats the freshly tuned best."
        )
        r1, r2, r3 = st.columns(3)
        with r1:
            lookback_bars = st.number_input(
                "Panel lookback (bars, 0 = full cleaned panel)",
                min_value=0,
                value=0,
                step=100,
                key="rt_panel_lb",
            )
            rt_expanding_pool = st.checkbox(
                "Expanding pool (cap = pool size; first origin = inner window)",
                value=True,
                help=(
                    "If on, refit uses rows [max(0, t−cap), t) with cap = history pool, so the first forecast is at index **inner window** "
                    "instead of **history pool** (fits “1000 rows total, 250 tune, 750 steps”). "
                    "If off, you need at least **history pool + 1** rows before the first origin."
                ),
                key="rt_expand_pool",
            )
            rt_history_pool = st.number_input(
                "History pool / cap (bars) for refit",
                min_value=120,
                value=1000,
                step=50,
                help="Fixed mode: exactly this many rows `[t − pool, t)`. Expanding mode: **maximum** history length (still capped at *t*).",
                key="rt_hist_pool",
            )
        with r2:
            rt_fit_window = st.number_input(
                "Inner window (bars) for tuning",
                min_value=40,
                value=250,
                step=10,
                help="Last N rows before *t* used for inner train/validation and grid scores.",
                key="rt_fit_w",
            )
            rt_train_frac_inner = st.slider(
                "Inner train fraction",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05,
                key="rt_train_frac_in",
            )
        with r3:
            rt_forecast_step = st.number_input(
                "Forecast step (bars between origins)",
                min_value=1,
                value=1,
                step=1,
                help="1 = predict every bar; 5 = every 5th bar (much faster).",
                key="rt_fc_step",
            )
            rt_max_steps = st.number_input(
                "Max forecast steps (0 = all origins)",
                min_value=0,
                value=250,
                step=50,
                help="Cap rolling iterations for speed; 0 runs every origin (can take hours).",
                key="rt_max_steps",
            )
            rt_family = st.selectbox(
                "Estimator grid",
                ["faster", "fast", "all", "logreg", "hgb", "rf", "xgb", "lgbm"],
                index=0,
                help=(
                    "**faster** = 3 small variants (fastest). **fast** = LogReg + HGB + RF grid (~15 fits/origin). "
                    "**all** adds XGB/LGBM (+slow)."
                ),
                key="rt_family",
            )
            rt_tune_n_jobs = st.number_input(
                "Parallel tune jobs",
                min_value=-1,
                max_value=32,
                value=-1,
                step=1,
                help=(
                    "How many CPU workers run the **parameter grid** at each forecast origin. "
                    "**-1** = auto (up to 16 cores). **1** = sequential (lowest CPU, slowest). "
                    "Requires joblib (installed with scikit-learn)."
                ),
                key="rt_tune_n_jobs",
            )
            rt_rolling_pca = st.number_input(
                "PCA components (rolling tune)",
                min_value=0,
                max_value=50,
                value=5,
                help="0 = raw features (impute + scale only). >0 = top-K PCs like walk-forward (fit on pool at deploy).",
                key="rt_rolling_pca",
            )
        with st.expander("Held-out inner test, sticky champion, rolling Sharpe blocks"):
            rt_inner_test_frac = st.slider(
                "Inner test fraction (0 = validation-only winner per family)",
                min_value=0.0,
                max_value=0.45,
                value=0.15,
                step=0.05,
                help=(
                    "Hold out this fraction of the **inner tuning window** as a final test set. "
                    "Among family winners, the single model with highest **test** accuracy is deployed."
                ),
                key="rt_inner_test_frac",
            )
            rt_sticky_champion = st.checkbox(
                "Sticky champion",
                value=False,
                help=(
                    "After inner test scoring: keep the **previous step’s** winning variant if its test accuracy is **≥** "
                    "the best newly tuned variant’s test accuracy."
                ),
                key="rt_sticky",
            )
            rt_rolling_chunk = st.number_input(
                "Rolling OOS block size (Sharpe distribution)",
                min_value=0,
                value=100,
                step=10,
                help="Split sequential strategy returns into chunks of this many bars and summarize Sharpe (etc.) per chunk (0 = skip).",
                key="rt_roll_chunk",
            )

    st.divider()
    st.subheader("Hold-out & walk-forward — model scope")
    st.caption(
        "Rolling tune does **not** use this list; narrow rolling runs with **Estimator grid** (e.g. `logreg`, `hgb`, `faster`)."
    )
    model_scope_sel = st.selectbox(
        "Fit only this model",
        ["All models"] + list(DIRECTION_MODEL_OPTIONS),
        index=0,
        help=(
            "Walk-forward and hold-out call the same sklearn / statsmodels suite. "
            "Pick one **exact** model name to speed up evaluation; ensembles need ≥2 member models fitted."
        ),
        key="eval_model_scope",
    )
    wf_models_filter = "ALL" if model_scope_sel == "All models" else str(model_scope_sel)

    run_btn = st.button("Run evaluation", type="primary")

    # --- Run evaluation (writes session_state snapshots; survives later widget reruns) ---
    if run_btn:
        if mode_wf:
            try:
                pack = _cached_walk_forward(
                    primary,
                    target_mode,
                    int(lead),
                    period,
                    int(study_bars),
                    ctx_key,
                    int(pca_components),
                    include_garch,
                    int(lookback_bars),
                    int(window_size),
                    float(train_frac),
                    int(step),
                    skip_lstm_wf,
                    target_basis_choice,
                    wf_models_filter,
                )
            except Exception as e:
                st.exception(e)
                return
            if pack.get("error"):
                st.error(str(pack.get("error")))
                return
            wf_out = pack["wf"]
            if wf_out.get("error"):
                st.error(f"Walk-forward: **{wf_out['error']}**")
                return
            base = pack["base"]
            st.session_state["wf_snapshot"] = {
                "base": base,
                "wf": wf_out,
                "meta": {
                    "lead": int(lead),
                    "pca_components": int(pca_components),
                    "target_mode": target_mode,
                    "models_filter": wf_models_filter,
                },
                "panel_before_lookback": pack.get("panel_before_lookback"),
            }
            st.session_state["holdout_snapshot"] = None
            st.session_state["last_base"] = base
            st.session_state["wf_oos_parts"] = wf_out.get("oos_parts", [])
            st.session_state["eval_mode"] = "wf"
            st.session_state["holdout_preds"] = None
            st.session_state["target_mode_snapshot"] = target_mode
            st.session_state["rolling_tune_snapshot"] = None
        elif mode_rt:
            try:
                pack_rt = _cached_rolling_tune(
                    primary,
                    target_mode,
                    int(lead),
                    period,
                    int(study_bars),
                    ctx_key,
                    include_garch,
                    int(lookback_bars),
                    int(rt_history_pool),
                    int(rt_fit_window),
                    float(rt_train_frac_inner),
                    int(rt_forecast_step),
                    int(rt_max_steps),
                    str(rt_family),
                    target_basis_choice,
                    bool(rt_expanding_pool),
                    float(rt_inner_test_frac),
                    bool(rt_sticky_champion),
                    int(rt_rolling_chunk),
                    int(rt_tune_n_jobs),
                    int(rt_rolling_pca),
                )
            except Exception as e:
                st.exception(e)
                return
            if pack_rt.get("error"):
                st.error(str(pack_rt.get("error")))
                return
            rt_out = pack_rt["rt"]
            if rt_out.get("error"):
                parts_err = [f"Rolling tune: **{rt_out['error']}**"]
                if rt_out.get("hint"):
                    parts_err.append(str(rt_out["hint"]))
                if rt_out.get("diagnostics"):
                    parts_err.append(f"Diagnostics: `{rt_out['diagnostics']}`")
                if rt_out.get("n_rows") is not None:
                    parts_err.append(
                        f"Rows in panel: **{rt_out['n_rows']}**, ``history_bars`` requested: **{rt_out.get('history_bars', '?')}**."
                    )
                st.error("\n\n".join(parts_err))
                return
            base_rt = pack_rt["base"]
            st.session_state["rolling_tune_snapshot"] = {
                "base": base_rt,
                "rt": rt_out,
                "meta": {
                    "lead": int(lead),
                    "target_mode": target_mode,
                    "history_pool": int(rt_history_pool),
                    "fit_window": int(rt_fit_window),
                    "expanding_pool": bool(rt_expanding_pool),
                    "inner_test_frac": float(rt_inner_test_frac),
                    "sticky_champion": bool(rt_sticky_champion),
                    "rolling_metrics_chunk": int(rt_rolling_chunk),
                    "tune_n_jobs": int(rt_tune_n_jobs),
                    "rolling_pca_components": int(rt_rolling_pca),
                },
                "panel_before_lookback": pack_rt.get("panel_before_lookback"),
            }
            st.session_state["wf_snapshot"] = None
            st.session_state["holdout_snapshot"] = None
            st.session_state["last_base"] = base_rt
            st.session_state["wf_oos_parts"] = rt_out.get("oos_parts", [])
            st.session_state["eval_mode"] = "rolling_tune"
            st.session_state["holdout_preds"] = None
            st.session_state["target_mode_snapshot"] = target_mode
        else:
            try:
                dset, res, preds = _cached_holdout_run(
                    primary,
                    target_mode,
                    int(lead),
                    period,
                    int(study_bars),
                    ctx_key,
                    int(pca_components),
                    include_garch,
                    target_basis_choice,
                    wf_models_filter,
                )
            except Exception as e:
                st.exception(e)
                return
            if dset.get("error"):
                st.error(f"Dataset: **{dset['error']}**")
                return
            st.session_state["holdout_snapshot"] = {
                "dset": dset,
                "res": res,
                "preds": preds,
                "meta": {
                    "lead": int(lead),
                    "target_mode": target_mode,
                    "models_filter": wf_models_filter,
                },
            }
            st.session_state["wf_snapshot"] = None
            st.session_state["rolling_tune_snapshot"] = None
            st.session_state["last_base"] = dset
            st.session_state["holdout_preds"] = preds
            st.session_state["eval_mode"] = "holdout"
            st.session_state["wf_oos_parts"] = []
            st.session_state["target_mode_snapshot"] = target_mode

    # --- Walk-forward results (show whenever we have a snapshot, not only on Run click) ---
    wf_snap = st.session_state.get("wf_snapshot")
    if wf_snap:
        base = wf_snap["base"]
        wf_out = wf_snap["wf"]
        meta = wf_snap.get("meta", {})
        lead_d = int(meta.get("lead", lead))
        pca_d = meta.get("pca_components", pca_components)
        tm_d = str(meta.get("target_mode", target_mode))
        wp = wf_out["window_params"]
        st.subheader("Walk-forward design")
        tlab = base.get("target_return_basis", "close")
        pre_lb = wf_snap.get("panel_before_lookback")
        pre_lb_md = ""
        if pre_lb:
            if pre_lb.get("trimmed"):
                pre_lb_md = (
                    f"- **Full cleaned panel (before lookback):** {pre_lb['n_rows_full']} rows, "
                    f"**{pre_lb['min_date_full']}** → **{pre_lb['max_date_full']}** "
                    f"(then **last {pre_lb['lookback_bars']}** rows only used for walk-forward)  \n"
                )
            else:
                pre_lb_md = (
                    f"- **Cleaned panel:** {pre_lb['n_rows_full']} rows, "
                    f"**{pre_lb['min_date_full']}** → **{pre_lb['max_date_full']}** "
                    f"(history length **{pre_lb['lookback_bars'] or 0}** → no trim)  \n"
                )
        mf_show = meta.get("models_filter", "ALL")
        st.markdown(
            f"{pre_lb_md}"
            f"- **Rows (after lookback, in WF):** {base['n_total']}  |  **Lookback setting:** {wp.get('lookback_bars') or 'full'}  \n"
            f"- **Models fit:** `{mf_show}` (``ALL`` = full suite)  \n"
            f"- **Target label:** `{tlab}` (``forward_r`` always close-to-close for backtests)  \n"
            f"- **Window:** {wp['window_size']} bars → train **{wp['train_n']}** / test **{wp['test_n']}**  |  "
            f"**Step:** {wp['step']}  |  **Folds:** {wp['n_folds']}  \n"
            f"- **Primary:** `{base['primary_id']}`  |  **Lead:** {lead_d}  |  **PCA k:** {pca_d or 'off'}"
        )

        st.subheader("Pooled OOS metrics (honest)")
        st.caption(
            "**Pooled** = concatenate all out-of-sample test rows across folds (non-overlapping when step = test size). "
            "**Fold summary** = simple mean of test metrics across folds (correlated folds — use pooled for headline)."
        )
        pdata = wf_out.get("pooled_oos_metrics")
        if pdata is not None and not pdata.empty:
            st.dataframe(pdata, use_container_width=True)

        with st.expander("Mean metrics across folds (reference)"):
            sm = wf_out.get("summary_by_fold")
            if sm is not None and not sm.empty:
                st.dataframe(sm, use_container_width=True)

        dl = wf_out.get("fold_results")
        if dl is not None and not dl.empty:
            st.download_button(
                "Download fold-level CSV",
                dl.to_csv(index=False).encode(),
                file_name=f"wf_folds_{base['primary_id']}_h{lead_d}_{tm_d}.csv",
                mime="text/csv",
            )

    rt_snap = st.session_state.get("rolling_tune_snapshot")
    if rt_snap:
        base_rt = rt_snap["base"]
        rt_out = rt_snap["rt"]
        meta_rt = rt_snap.get("meta", {})
        lead_rt = int(meta_rt.get("lead", lead))
        tm_rt = str(meta_rt.get("target_mode", target_mode))
        pp = rt_out.get("params", {})
        st.subheader("Rolling tune — sequential 1-step OOS")
        st.markdown(
            f"- **Panel:** `{base_rt['primary_id']}`  |  **Lead:** {lead_rt}  |  **Target:** `{tm_rt}`  \n"
            f"- **History pool / cap:** {pp.get('history_bars')} bars  |  **Expanding pool:** {pp.get('expanding_pool')}  \n"
            f"- **Inner tuning window:** {pp.get('fit_window_bars')} bars  |  **Inner split:** "
            f"{pp.get('inner_train_n')} train / {pp.get('inner_val_n')} val / {pp.get('inner_test_n', 0)} test  \n"
            f"- **Inner test fraction:** {pp.get('inner_test_frac', 0)}  |  **Sticky champion:** {pp.get('sticky_champion')}  |  "
            f"**Rolling block size:** {pp.get('rolling_metrics_chunk')}  \n"
            f"- **Forecast step:** {pp.get('step')}  |  **Origins run:** {pp.get('n_forecast_origins')}  \n"
            f"- **Estimator grid:** `{pp.get('family_filter')}`  |  **Parallel tune jobs:** "
            f"{pp.get('tune_n_jobs_requested', '—')} (effective {pp.get('tune_n_jobs_effective', '—')})  |  "
            f"**Rolling PCA *k*:** {pp.get('global_pca_components', 'off')}"
        )
        rb_sum = rt_out.get("rolling_block_summary_by_model") or {}
        if rb_sum:
            st.markdown("**Rolling OOS block stats** (mean / std / min / max of each block’s Sharpe, return, etc.; blocks are chronological).")
            for mname, summary in rb_sum.items():
                sharp = summary.get("sharpe", {})
                if sharp.get("n_blocks", 0):
                    st.caption(f"Model: `{mname}` — blocks: {int(sharp.get('n_blocks', 0))}")
                    st.json({"sharpe": sharp, "ann_return": summary.get("ann_return", {})})
        summ_rt = rt_out.get("summary_by_model")
        if summ_rt is not None and not summ_rt.empty:
            st.markdown("**Mean sequential OOS accuracy** (hit rate of next-bar predictions).")
            st.dataframe(summ_rt, use_container_width=True)
        ps = rt_out.get("per_step_predictions")
        if ps is not None and not ps.empty:
            with st.expander("Per-step predictions (detail)", expanded=False):
                st.caption("Each row: one tuned model at one forecast origin. Download for full history.")
                st.dataframe(ps, use_container_width=True, height=min(420, 28 * (min(len(ps), 400) + 2)))
            st.download_button(
                "Download per-step predictions CSV",
                ps.to_csv(index=False).encode(),
                file_name=f"rolling_tune_steps_{base_rt['primary_id']}_h{lead_rt}_{tm_rt}.csv",
                mime="text/csv",
            )

    # --- Hold-out results ---
    ho_snap = st.session_state.get("holdout_snapshot")
    if ho_snap:
        dset = ho_snap["dset"]
        res = ho_snap["res"]
        meta_h = ho_snap.get("meta", {})
        lead_h = int(meta_h.get("lead", lead))
        tm_h = str(meta_h.get("target_mode", target_mode))

        st.subheader("Sample design (hold-out)")
        tlab = dset.get("target_return_basis", "close")
        mf_h = meta_h.get("models_filter", "ALL")
        st.markdown(
            f"- **Rows:** {dset['n_total']}  |  **Train / test:** {dset['n_train']} / {dset['n_test']}  |  "
            f"**Features:** {dset['n_features']}  \n"
            f"- **Models fit:** `{mf_h}` (``ALL`` = full suite)  \n"
            f"- **Target label:** `{tlab}` (``forward_r`` = close-to-close; use data viewer for `hi_exc` / `lo_exc` when H/L)  \n"
            f"- **Primary:** `{dset['primary_id']}`  |  **Lead:** {lead_h}"
        )

        if res.empty:
            st.warning("No model results.")
        else:
            display_cols = [
                "model",
                "pair",
                "lead",
                "target",
                "n_train",
                "n_test",
                "n_features",
                "train_accuracy",
                "test_accuracy",
                "train_balanced_acc",
                "test_balanced_acc",
                "train_log_loss",
                "test_log_loss",
                "train_f1_macro",
                "test_f1_macro",
                "train_roc_auc",
                "test_roc_auc",
            ]
            extra_cols = [c for c in res.columns if c.startswith("extra_")]
            show = res[[c for c in display_cols + extra_cols if c in res.columns]]
            st.subheader("Hold-out metrics")
            st.dataframe(show, use_container_width=True, height=min(640, 32 * (len(show) + 2)))
            st.download_button(
                "Download results CSV",
                show.to_csv(index=False).encode(),
                file_name=f"direction_holdout_{dset['primary_id']}_h{lead_h}_{tm_h}.csv",
                mime="text/csv",
            )

    # --- View data (both modes) ---
    base_view = st.session_state.get("last_base")
    if base_view and not base_view.get("error"):
        with st.expander("View feature matrix (labeled dates)", expanded=False):
            prev = _data_preview_table(base_view)
            if prev is not None:
                wf_snap_vm = st.session_state.get("wf_snapshot")
                rt_snap_vm = st.session_state.get("rolling_tune_snapshot")
                pre_lb_vm = None
                if wf_snap_vm:
                    pre_lb_vm = wf_snap_vm.get("panel_before_lookback")
                elif rt_snap_vm:
                    pre_lb_vm = rt_snap_vm.get("panel_before_lookback")
                if pre_lb_vm and pre_lb_vm.get("trimmed"):
                    st.warning(
                        f"Walk-forward **trimmed** the panel with **History length = {pre_lb_vm['lookback_bars']}**: "
                        f"you only have the **last {len(prev):,}** rows here (through **{prev['date'].max()}**). "
                        f"Before trim, the cleaned matrix had **{pre_lb_vm['n_rows_full']:,}** rows through "
                        f"**{pre_lb_vm['max_date_full']}**. "
                        "Set **History length (bars)** to **0** and **Run evaluation** again to load the **full** cleaned panel."
                    )
                preview_choice = st.radio(
                    "Preview slice",
                    ["Most recent 500 rows (default)", "Oldest 500 rows"],
                    horizontal=True,
                    key="feature_matrix_preview_slice",
                    help="Rows are in calendar order. **Oldest** was the old default and hides recent years in long histories.",
                )
                slice_df = (
                    prev.tail(500) if preview_choice.startswith("Most recent") else prev.head(500)
                )
                st.dataframe(slice_df, use_container_width=True, height=400)
                st.caption(
                    f"Preview window: **{slice_df['date'].min()}** → **{slice_df['date'].max()}** ({len(slice_df)} rows). "
                    f"Full panel in memory: **{prev['date'].min()}** → **{prev['date'].max()}** ({len(prev):,} rows). "
                    "Columns: `date`, `y`, close-to-close **forward_r**, optional **hi_exc** / **lo_exc**, plus features."
                )
                fx_m = _cached_fxcm_meta(primary, fxcm_primary_parquet_mtime(primary))
                if fx_m.get("end"):
                    fx_end = pd.to_datetime(fx_m["end"]).normalize()
                    p_end = pd.to_datetime(prev["date"].max()).normalize()
                    if fx_end > p_end + pd.Timedelta(days=21):
                        st.error(
                            f"FXCM cache for this pair has daily bars through **{fx_end.date()}**, but the **feature matrix** "
                            f"ends **{p_end.date()}**. Recent history was dropped when building features "
                            "(merge / NaN thresholds after joining FRED and other pairs). "
                            "Try fewer **Other pairs**, **study_bars = 0**, refresh **FRED** in the Currencies app, then **Run evaluation** again."
                        )
                st.download_button(
                    "Download feature CSV (full)",
                    prev.to_csv().encode(),
                    file_name=f"features_{base_view.get('primary_id', 'panel')}.csv",
                    mime="text/csv",
                )

    # --- Strategy backtest ---
    st.subheader("Strategy backtest (probability rules)")
    base_bt = st.session_state.get("last_base")
    pmin, pmax = _panel_calendar_range(base_bt if isinstance(base_bt, dict) else None)
    if pmin is not None and pmax is not None:
        st.caption(
            f"**Feature / label panel (current run, after lookback trim):** **{pmin.date()}** → **{pmax.date()}**. "
            "The equity curve uses **pooled OOS** predictions vs **forward returns** on those rows—see dates after you run charts."
        )
    st.caption(
        "**Binary:** long if class 1 wins else short; **threshold** mode requires max class probability ≥ threshold or flat. "
        "**Ternary:** long class 2, short class 0, flat if middle class wins or below threshold."
    )
    oos_parts_bt: list = []
    if st.session_state.get("eval_mode") == "holdout" and st.session_state.get("holdout_preds"):
        dlast = st.session_state.get("last_base")
        if dlast and not dlast.get("error"):
            oos_parts_bt = holdout_oos_parts(dlast, st.session_state["holdout_preds"])
    elif st.session_state.get("wf_oos_parts"):
        oos_parts_bt = st.session_state["wf_oos_parts"]

    tm_bt = str(st.session_state.get("target_mode_snapshot", target_mode))

    models_avail: list[str] = []
    if oos_parts_bt:
        models_avail = sorted({k for b in oos_parts_bt for k in b.get("preds", {}).keys()})
    pick = st.multiselect(
        "Models to chart",
        models_avail,
        default=models_avail[: min(3, len(models_avail))] if models_avail else [],
        key="bt_models_ms",
    )
    trade_mode_ui = st.radio(
        "Trading mode",
        ["Mode 1 (always in)", "Mode 2 (probability threshold)"],
        horizontal=True,
        key="bt_trade_mode",
    )
    thr = st.slider(
        "Min max-class probability (mode 2)",
        min_value=0.34,
        max_value=0.99,
        value=0.51,
        step=0.01,
        key="bt_thr_slider",
    )
    oos_tail_n = st.number_input(
        "Plot only last N OOS rows per model (0 = full pooled history)",
        min_value=0,
        max_value=500_000,
        value=0,
        step=50,
        key="bt_oos_tail",
        help="Zoom to recent out-of-sample periods without changing the underlying evaluation.",
    )

    if pick and oos_parts_bt and st.button("Run backtest charts", key="bt_run_btn"):
        _run_backtest_charts(
            oos_parts_bt,
            pick,
            tm_bt,
            trade_mode_ui,
            float(thr),
            oos_tail=int(oos_tail_n),
            sequential_oos=st.session_state.get("eval_mode") == "rolling_tune",
        )


if __name__ == "__main__":
    if _running_inside_streamlit():
        main()
    else:
        import subprocess

        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
                cwd=str(_ROOT),
            )
        )
