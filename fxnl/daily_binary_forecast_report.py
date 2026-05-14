"""Daily binary FX forecast report from the completed rolling-batch Excel workbook.

Filters the historical Excel report to binary runs with average OOS accuracy >= threshold, refits the
latest winning batch model on the latest labeled history, forecasts the newest feature row, writes an
Excel report, and optionally emails it via Gmail SMTP.

Gmail credentials:
  set GMAIL_USER=your_sender@gmail.com
  set GMAIL_APP_PASSWORD=your_gmail_app_password
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxnl.data_panel import _primary_high_low_close, prepare_direction_frame
from fxnl.paths import patch_currencies_sys_path
from fxnl.qualified_models import load_qualified_models
from fxnl.rolling_tune_oos import _grid_variants_for_family, _prep_deploy_full_one
from fxnl.targets import (
    binary_labels,
    binary_labels_high_low,
    forward_simple_return,
    forward_window_excursions,
)

logger = logging.getLogger(__name__)


def _context_all(primary: str, all_pairs: list[str]) -> list[str]:
    return [primary, *[x for x in all_pairs if x != primary]]


def _binary_sharpe(steps: pd.DataFrame) -> tuple[float, int]:
    """Binary strategy: pred 1 = long, pred 0 = short; return = position * forward_r."""
    if steps.empty:
        return float("nan"), 0
    pred = pd.to_numeric(steps["pred_class"], errors="coerce")
    r = pd.to_numeric(steps["forward_r"], errors="coerce")
    pos = np.where(pred >= 0.5, 1.0, -1.0)
    sr = pd.Series(pos * r.to_numpy(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    if sr.empty:
        return float("nan"), 0
    sd = float(sr.std(ddof=1))
    sh = float((sr.mean() / sd) * np.sqrt(252.0)) if sd > 1e-15 else float("nan")
    return sh, int(len(sr))


def _send_email_with_attachment(
    *,
    to_addr: str,
    subject: str,
    body: str,
    attachment: Path,
) -> None:
    user = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not user or not password:
        raise RuntimeError("Set GMAIL_USER and GMAIL_APP_PASSWORD before using --send-email")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(body)
    data = attachment.read_bytes()
    msg.add_attachment(
        data,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=attachment.name,
    )
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)


def _recommendation(pred_class: int, acc: float) -> tuple[str, str]:
    side = "Buy" if int(pred_class) == 1 else "Sell"
    if acc >= 0.65:
        side = f"Strong {side}"
    return side, "long" if int(pred_class) == 1 else "short"


# Per-run cache: build the heavy data panel ONCE per primary across all (lead, basis)
# combos in a single ``build_daily_report`` call. Cleared at the start of every run so
# successive invocations always pick up fresh data and (with --inject-live-quote) the
# latest live spot.
_PANEL_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, Any] | None] = {}


def _clear_panel_cache() -> None:
    _PANEL_CACHE.clear()


def _cached_features_panel(primary: str, all_pairs: list[str]) -> dict[str, Any] | None:
    """Build the cross-pair feature panel once per ``(primary, sorted_pairs)``."""
    key = (primary, tuple(sorted(all_pairs)))
    if key in _PANEL_CACHE:
        return _PANEL_CACHE[key]

    base = prepare_direction_frame(
        primary,
        context_instruments=_context_all(primary, list(all_pairs)),
        period="max",
        study_bars=None,
        lead=1,
        target_mode="binary",
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=True,
        target_return_basis="close",
        keep_unlabeled_tail=True,
    )

    if base.get("error"):
        _PANEL_CACHE[key] = {"error": str(base["error"])}
        return _PANEL_CACHE[key]

    X = base["X"].copy().sort_index()
    close = base.get("close")
    if not isinstance(close, pd.Series):
        _PANEL_CACHE[key] = {"error": "panel_missing_close_series"}
        return _PANEL_CACHE[key]
    close = close.reindex(X.index).astype(float)

    try:
        hi_lo = _primary_high_low_close(primary, "max", None, X.index)
    except Exception as e:
        logger.warning("hi/lo fetch failed for %s: %s", primary, e)
        hi_lo = None

    _PANEL_CACHE[key] = {
        "X": X,
        "close": close,
        "hi_lo_close": hi_lo,
        "error": "",
    }
    return _PANEL_CACHE[key]


def _forecast_one(
    *,
    primary: str,
    all_pairs: list[str],
    lead: int,
    target_basis: str,
    chosen_variant: str,
    lookback_bars: int,
    pca_k: int,
    family_filter: str,
    align_to_next_bar: bool = False,
) -> dict[str, Any]:
    """Refit `chosen_variant` on the latest labeled history and forecast one bar.

    When ``align_to_next_bar`` is True the input feature row is taken at index
    ``n - lead`` (n = panel length with the unlabeled tail), so a lead-L model
    forecasts the **next bar after the panel end** — i.e. every lead targets the
    same forecast date ("tomorrow"). Otherwise the latest available feature row
    (index -1) is used and lead-L forecasts ``panel_end + L`` bars.

    The heavy panel build is cached per primary (see ``_cached_features_panel``);
    here we just compute labels for this ``(lead, target_basis)`` and refit.
    """
    panel = _cached_features_panel(primary, list(all_pairs))
    if panel is None or panel.get("error"):
        return {"forecast_error": str((panel or {}).get("error") or "panel_build_failed")}

    X: pd.DataFrame = panel["X"]
    close: pd.Series = panel["close"]
    hi_lo = panel.get("hi_lo_close")

    L = int(lead)
    tb = str(target_basis or "close").strip().lower()

    r = forward_simple_return(close, L)
    if tb in ("high_low", "hl", "path"):
        if hi_lo is None:
            return {"forecast_error": "high_low_data_missing_for_primary"}
        high_s, low_s, close_s = hi_lo
        exc = forward_window_excursions(high_s, low_s, close_s, L)
        y = binary_labels_high_low(exc, tie_break_close=True)
    else:
        y = binary_labels(r)

    y = y.reindex(X.index)
    r = r.reindex(X.index)

    labeled_mask = y.notna() & r.notna()
    if int(labeled_mask.sum()) < 80:
        return {"forecast_error": f"too_few_labeled_rows:{int(labeled_mask.sum())}"}

    X_labeled = X.loc[labeled_mask]
    y_labeled = y.loc[labeled_mask].astype(int)
    if len(X_labeled) > int(lookback_bars):
        X_labeled = X_labeled.iloc[-int(lookback_bars) :]
        y_labeled = y_labeled.loc[X_labeled.index]

    n = len(X)
    L = int(lead)
    if align_to_next_bar:
        if n < L + 1:
            return {"forecast_error": f"panel_too_short_for_lead:n={n},lead={L}"}
        input_idx = n - L
    else:
        input_idx = n - 1
    X_live = X.iloc[[input_idx]]
    latest_feature_date = pd.Timestamp(X.index[-1])
    input_feature_date = pd.Timestamp(X.index[input_idx])
    latest_labeled_date = pd.Timestamp(X_labeled.index[-1])
    if align_to_next_bar:
        latest_norm = latest_feature_date.normalize()
        model_target = (latest_norm + pd.tseries.offsets.BDay(1)).normalize()
        ny_today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        calendar_next = (ny_today + pd.tseries.offsets.BDay(1)).normalize()
        target_forecast_date = max(model_target, calendar_next)
    else:
        target_forecast_date = (latest_feature_date + pd.tseries.offsets.BDay(L)).normalize()
    latest_panel_close = float(close.iloc[-1]) if isinstance(close, pd.Series) and pd.notna(close.iloc[-1]) else float("nan")
    input_close = (
        float(close.iloc[input_idx]) if isinstance(close, pd.Series) and pd.notna(close.iloc[input_idx]) else float("nan")
    )

    labels = np.sort(np.unique(y_labeled.to_numpy(dtype=int)))
    if len(labels) < 2:
        return {"forecast_error": "training_labels_single_class"}
    to_idx = {int(v): i for i, v in enumerate(labels)}
    y_map = np.array([to_idx[int(v)] for v in y_labeled.to_numpy(dtype=int)], dtype=int)
    n_cls = int(len(labels))

    fn = None
    for dn, _ks, fit_fn in _grid_variants_for_family(family_filter, n_cls, rf_n_jobs=-1):
        if dn == chosen_variant:
            fn = fit_fn
            break
    if fn is None:
        return {"forecast_error": f"chosen_variant_not_in_grid:{chosen_variant}"}

    Xfull, Xpred = _prep_deploy_full_one(
        np.asarray(X_labeled, dtype=float),
        np.asarray(X_live, dtype=float),
        int(pca_k) if int(pca_k) > 0 else None,
        42,
    )
    proba, _pred, _model = fn(Xfull, y_map, Xpred)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    pred_idx = int(np.argmax(proba, axis=1)[0])
    # labels are 0/1 for binary; fall back to inverse map when needed.
    inv = {v: k for k, v in to_idx.items()}
    pred_raw = int(inv.get(pred_idx, pred_idx))
    p_sell = float(proba[0, to_idx.get(0, 0)]) if 0 in to_idx else float("nan")
    p_buy = float(proba[0, to_idx.get(1, min(1, proba.shape[1] - 1))]) if 1 in to_idx else float("nan")

    return {
        "latest_feature_date": latest_feature_date.date().isoformat(),
        "latest_labeled_date": latest_labeled_date.date().isoformat(),
        "input_feature_date": input_feature_date.date().isoformat(),
        "target_forecast_date": target_forecast_date.date().isoformat(),
        "latest_panel_close": latest_panel_close,
        "input_close": input_close,
        "pred_class": pred_raw,
        "p_sell": p_sell,
        "p_buy": p_buy,
        "forecast_error": "",
    }


def build_daily_report(
    *,
    source_xlsx: Path,
    out_xlsx: Path,
    min_accuracy: float,
    strong_accuracy: float,
    lookback_bars: int,
    pca_k: int,
    family_filter: str,
    align_to_next_bar: bool = False,
    qualified_json: Path | None = None,
) -> Path:
    patch_currencies_sys_path()
    try:
        from src.config import FX_PAIR_INSTRUMENTS  # noqa: PLC0415
    except Exception as e:
        raise RuntimeError(f"Could not import FX_PAIR_INSTRUMENTS: {e}") from e

    _clear_panel_cache()
    run_summary = pd.read_excel(source_xlsx, sheet_name="run_summary")
    all_batches = pd.read_excel(source_xlsx, sheet_name="all_batches")
    all_steps = pd.read_excel(source_xlsx, sheet_name="all_steps")

    rs = run_summary.copy()
    rs["error"] = rs["error"].fillna("")
    rs = rs[
        (rs["error"].astype(str).str.len() == 0)
        & (rs["target_mode"].astype(str).str.lower() == "binary")
        & (pd.to_numeric(rs["mean_seq_oos_acc"], errors="coerce") >= float(min_accuracy))
    ].copy()

    qualified_run_ids: set[str] | None = None
    if qualified_json is not None:
        qrows = load_qualified_models(Path(qualified_json))
        if qrows:
            qualified_run_ids = {str(q.get("run_id")) for q in qrows if q.get("run_id")}
            before = len(rs)
            rs = rs[rs["run_id"].astype(str).isin(qualified_run_ids)].copy()
            logger.info(
                "Endorsed-models gate: %d/%d run_ids retained from %s",
                len(rs),
                before,
                Path(qualified_json).name,
            )
        else:
            logger.warning(
                "qualified_json %s missing or empty; falling back to min-accuracy gate only.",
                qualified_json,
            )

    latest_batches = (
        all_batches.sort_values(["run_id", "batch_idx"])
        .groupby("run_id", as_index=False)
        .tail(1)[["run_id", "chosen_variant", "chosen_inner_test_acc", "same_model_as_previous_batch"]]
    )
    rs = rs.merge(latest_batches, on="run_id", how="left")

    rows: list[dict[str, Any]] = []
    all_pairs = list(FX_PAIR_INSTRUMENTS)
    total = len(rs)
    for i, (_, row) in enumerate(rs.iterrows(), start=1):
        rid = str(row["run_id"])
        steps = all_steps[all_steps["run_id"].astype(str) == rid]
        sh, n_strat = _binary_sharpe(steps)
        chosen_variant = str(row.get("chosen_variant", "")).strip()
        logger.info(
            "Forecast %d/%d %s lead=%s basis=%s variant=%s",
            i,
            total,
            row["primary"],
            int(row["lead"]),
            str(row["target_return_basis"]),
            chosen_variant,
        )
        fc = _forecast_one(
            primary=str(row["primary"]),
            all_pairs=all_pairs,
            lead=int(row["lead"]),
            target_basis=str(row["target_return_basis"]),
            chosen_variant=chosen_variant,
            lookback_bars=int(lookback_bars),
            pca_k=int(pca_k),
            family_filter=family_filter,
            align_to_next_bar=align_to_next_bar,
        )
        acc = float(row["mean_seq_oos_acc"])
        pred_class = fc.get("pred_class")
        if fc.get("forecast_error") or pred_class is None:
            signal, position = "", ""
        else:
            signal, position = _recommendation(int(pred_class), acc)
            if acc < float(strong_accuracy) and signal.startswith("Strong "):
                signal = signal.replace("Strong ", "")

        rows.append(
            {
                "primary": row["primary"],
                "lead_days": int(row["lead"]),
                "target_return_basis": row["target_return_basis"],
                "chosen_variant": chosen_variant,
                "avg_oos_accuracy": acc,
                "binary_oos_sharpe": sh,
                "binary_oos_rows_for_sharpe": n_strat,
                "global_batch_stability_rate": row.get("global_batch_stability_rate"),
                "latest_inner_test_acc": row.get("chosen_inner_test_acc"),
                "same_as_previous_batch": row.get("same_model_as_previous_batch"),
                "latest_feature_date": fc.get("latest_feature_date"),
                "latest_labeled_date": fc.get("latest_labeled_date"),
                "input_feature_date": fc.get("input_feature_date"),
                "target_forecast_date": fc.get("target_forecast_date"),
                "latest_panel_close": fc.get("latest_panel_close"),
                "input_close": fc.get("input_close"),
                "pred_class": pred_class,
                "p_sell": fc.get("p_sell"),
                "p_buy": fc.get("p_buy"),
                "position": position,
                "recommendation": signal,
                "forecast_error": fc.get("forecast_error", ""),
            },
        )

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            ["avg_oos_accuracy", "binary_oos_sharpe"],
            ascending=[False, False],
            na_position="last",
        )

    meta = pd.DataFrame(
        [
            {
                "source_xlsx": str(source_xlsx),
                "min_accuracy": min_accuracy,
                "strong_accuracy": strong_accuracy,
                "lookback_bars": lookback_bars,
                "pca_k": pca_k,
                "family_filter": family_filter,
                "align_to_next_bar": bool(align_to_next_bar),
                "note": (
                    "Binary only. pred_class 1=Buy, 0=Sell. "
                    "With align_to_next_bar, each model row targets the next session after the feature panel "
                    "(use after daily data is complete, e.g. post US close). "
                    "Optional --inject-live-quote appends/refreshes today's synthetic bar from Yahoo when FXNL_INJECT_LIVE_QUOTE=1."
                ),
            },
        ],
    )
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="daily_signals", index=False)
        meta.to_excel(writer, sheet_name="metadata", index=False)
    return out_xlsx


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build/email daily binary FX forecast report.")
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--source-xlsx", type=Path, default=root / "batch_hold_full_export.xlsx")
    p.add_argument("--out-xlsx", type=Path, default=root / "daily_binary_fx_forecast.xlsx")
    p.add_argument("--to", default="mr.mh.rahmani@gmail.com")
    p.add_argument("--min-accuracy", type=float, default=0.60)
    p.add_argument("--strong-accuracy", type=float, default=0.65)
    p.add_argument("--lookback-bars", type=int, default=1000)
    p.add_argument("--pca", type=int, default=5)
    p.add_argument("--family-filter", default="faster")
    p.add_argument("--send-email", action="store_true")
    p.add_argument(
        "--align-to-next-bar",
        action="store_true",
        help=(
            "Use input row at index n-L for a lead-L model so every lead's forecast "
            "targets the same next bar after the panel end (i.e., 'tomorrow')."
        ),
    )
    p.add_argument(
        "--qualified-json",
        type=Path,
        default=root / "qualified_models.json",
        help=(
            "Endorsed model list produced by ``fxnl.qualified_models``. "
            "If present, only its run_ids are forecast (skipping any other combos). "
            "Pass an empty/non-existent path to disable."
        ),
    )
    p.add_argument(
        "--inject-live-quote",
        action="store_true",
        help=(
            "Pull today's Yahoo intraday for each FX pair and append it as a synthetic "
            "today-bar (Open=first 1m, High=session max, Low=session min, Close=live spot) "
            "before building the panel. With --align-to-next-bar, every lead then targets "
            "the next FX trading day after today's live close."
        ),
    )
    args = p.parse_args(argv)
    if args.inject_live_quote:
        os.environ["FXNL_INJECT_LIVE_QUOTE"] = "1"

    out = build_daily_report(
        source_xlsx=args.source_xlsx,
        out_xlsx=args.out_xlsx,
        min_accuracy=float(args.min_accuracy),
        strong_accuracy=float(args.strong_accuracy),
        lookback_bars=int(args.lookback_bars),
        pca_k=int(args.pca),
        family_filter=str(args.family_filter),
        align_to_next_bar=bool(args.align_to_next_bar),
        qualified_json=args.qualified_json if args.qualified_json and args.qualified_json.exists() else None,
    )
    logging.info("Wrote %s", out.resolve())

    if args.send_email:
        _send_email_with_attachment(
            to_addr=str(args.to),
            subject="Daily FX Binary Forecast Report",
            body=(
                "Attached is the daily binary FX forecast report. "
                "Signals are filtered to average OOS accuracy >= 60%; >=65% is marked Strong."
            ),
            attachment=out,
        )
        logging.info("Sent report to %s", args.to)


if __name__ == "__main__":
    main()
