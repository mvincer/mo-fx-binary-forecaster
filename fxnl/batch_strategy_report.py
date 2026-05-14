"""
Batch walk-forward + threshold backtest across all FX pairs → PDF summary.

Designed for offline runs (not Streamlit). Uses the same panel prep and walk-forward
stack as the Direction app.
"""

from __future__ import annotations

import argparse
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxnl.backtest import (
    flatten_rolling_summary,
    oos_rolling_block_stats,
    performance_stats,
    stitched_oos_frame,
    strategy_returns_binary,
    strategy_returns_ternary,
    summarize_rolling_blocks,
)
from fxnl.data_panel import prepare_direction_frame
from fxnl.paths import patch_currencies_sys_path
from fxnl.walk_forward import walk_forward_evaluate

logger = logging.getLogger(__name__)


def apply_end_date_last_n(
    X: pd.DataFrame,
    y: pd.Series,
    forward_r: pd.Series,
    end_date: str | pd.Timestamp,
    n: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    """Keep rows with index <= ``end_date``, then take the last ``n`` rows (if enough data)."""
    end = pd.Timestamp(end_date).normalize()
    X = X.sort_index()
    y = y.reindex(X.index)
    forward_r = forward_r.reindex(X.index)
    mask = pd.DatetimeIndex(pd.to_datetime(X.index)).normalize() <= end
    X = X.loc[mask]
    y = y.loc[X.index]
    forward_r = forward_r.loc[X.index]
    eff_end = X.index.max() if len(X) else end
    if len(X) > n:
        X = X.iloc[-n:]
        y = y.loc[X.index]
        forward_r = forward_r.loc[X.index]
    return X, y, forward_r, pd.Timestamp(eff_end)


def run_one_pair(
    primary: str,
    *,
    context_instruments: list[str],
    end_date: str,
    n_tail: int,
    window_size: int,
    train_frac: float,
    pca_components: int,
    prob_threshold: float,
    lead: int = 1,
    target_mode: str = "binary",
    include_garch: bool = True,
    rolling_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Walk-forward OOS + stitched strategy backtest at probability threshold (binary or ternary)."""
    tm = str(target_mode).strip().lower()
    base = prepare_direction_frame(
        primary,
        context_instruments=context_instruments,
        period="max",
        study_bars=None,
        lead=lead,
        target_mode=tm,
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=include_garch,
        target_return_basis="close",
    )
    out: dict[str, Any] = {
        "primary": primary,
        "primary_id": base.get("primary_id", ""),
        "target_mode": tm,
        "lead": int(lead),
    }
    if base.get("error"):
        out["error"] = str(base["error"])
        return out

    X, y, forward_r = base["X"], base["y"], base["forward_r"]
    X, y, forward_r, eff_end = apply_end_date_last_n(X, y, forward_r, end_date, n_tail)
    out["effective_end_date"] = str(eff_end.date())
    out["n_rows_used"] = len(X)

    if len(X) < window_size:
        out["error"] = f"too_few_rows:{len(X)} for window {window_size}"
        return out

    wf = walk_forward_evaluate(
        X,
        y,
        forward_r,
        window_size=int(window_size),
        train_frac=float(train_frac),
        step=None,
        lookback_bars=None,
        random_state=42,
        global_pca_components=int(pca_components) if pca_components > 0 else None,
        skip_lstm=True,
        pair=str(base.get("primary_id", primary)),
        lead=int(lead),
        target_mode=tm,
    )
    if wf.get("error"):
        out["error"] = str(wf["error"])
        return out

    pooled = wf.get("pooled_oos_metrics")
    if pooled is None or pooled.empty:
        out["error"] = "no_pooled_metrics"
        return out

    oos_parts = wf.get("oos_parts", [])
    pool_acc = pooled.set_index("model")["pooled_test_accuracy"].to_dict()
    rows: list[dict[str, Any]] = []
    for mname in pooled["model"].unique():
        try:
            f = stitched_oos_frame(oos_parts, str(mname))
            if f is None or f.empty:
                continue
            if tm == "binary":
                if "p_0" not in f.columns or "p_1" not in f.columns:
                    continue
                sr, pos = strategy_returns_binary(
                    f,
                    mode="threshold",
                    prob_threshold=float(prob_threshold),
                )
            else:
                if not all(c in f.columns for c in ("p_0", "p_1", "p_2")):
                    continue
                sr, pos = strategy_returns_ternary(
                    f,
                    mode="threshold",
                    prob_threshold=float(prob_threshold),
                )
            st = performance_stats(sr, positions=pos)
            row: dict[str, Any] = {
                "model": str(mname),
                "pooled_test_accuracy": float(pool_acc.get(mname, float("nan"))),
                "sharpe": st["sharpe"],
                "ann_return": st["ann_return"],
                "max_drawdown": st["max_drawdown"],
                "sortino": st["sortino"],
                "hit_rate": st["hit_rate"],
                "pct_traded": st["pct_traded"],
                "n_oos_days": st["n_days"],
            }
            if rolling_chunk_size is not None and int(rolling_chunk_size) > 0:
                blocks, rmeta = oos_rolling_block_stats(
                    sr,
                    pos,
                    chunk_size=int(rolling_chunk_size),
                )
                summ = summarize_rolling_blocks(blocks)
                row["rolling_oos_rows"] = rmeta.get("n_oos_rows")
                row["rolling_n_chunks"] = rmeta.get("n_chunks")
                row["rolling_note"] = str(rmeta.get("note") or "")
                row["rolling_block_size"] = int(rolling_chunk_size)
                row.update(flatten_rolling_summary("roll_", summ))
            rows.append(row)
        except Exception as e:
            logger.debug("backtest skip %s %s: %s", primary, mname, e)
            continue

    if not rows:
        out["error"] = "no_backtest_rows"
        return out

    df_bt = pd.DataFrame(rows)
    df_bt["_skey"] = df_bt["sharpe"].apply(
        lambda x: float(x) if not (isinstance(x, float) and math.isnan(x)) else -1e9
    )
    df_bt = df_bt.sort_values("_skey", ascending=False).drop(columns="_skey")

    wp = wf.get("window_params", {})
    out.update(
        {
            "error": None,
            "window_params": wp,
            "pooled_oos_metrics": pooled,
            "backtest_table": df_bt,
            "top_models": df_bt.head(5)["model"].tolist(),
            "oos_span": {
                "n_folds": wp.get("n_folds"),
                "train_n": wp.get("train_n"),
                "test_n": wp.get("test_n"),
            },
        },
    )
    return out


def _pdf_txt(s: Any) -> str:
    """fpdf2 Helvetica only supports latin-1; strip problematic unicode."""
    t = str(s) if s is not None else ""
    for a, b in (
        ("\u2014", "-"),
        ("\u2013", "-"),
        ("\u2212", "-"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2019", "'"),
    ):
        t = t.replace(a, b)
    return t.encode("latin-1", errors="replace").decode("latin-1")


def _fmt_num(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
        if math.isnan(xf):
            return "NA"
        return f"{xf:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_pdf(
    results: list[dict[str, Any]],
    out_path: Path,
    *,
    params: dict[str, Any],
) -> None:
    """Write multi-page PDF using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise RuntimeError("Install fpdf2: pip install fpdf2") from e

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _pdf_txt("FX Direction - Strategy & Backtest Report"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(4)
    for k, v in params.items():
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 6, _pdf_txt(f"{k}: {v}"))
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        pdf.epw,
        5,
        _pdf_txt(
            "Walk-forward: rolling window with inner train/test split; OOS predictions pooled. "
            "Backtest: binary = long/short from two classes; ternary = long / flat / short from three classes. "
            "Threshold mode: trade only when max class probability meets the threshold (else flat). "
            "Rankings: Sharpe ratio on stitched OOS strategy returns."
        ),
    )

    ok = [r for r in results if not r.get("error")]
    bad = [r for r in results if r.get("error")]
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_txt("Run summary"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _pdf_txt(f"Successful runs: {len(ok)} / {len(results)}"), ln=1)
    if bad:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, _pdf_txt("Skipped / errors:"), ln=1)
        pdf.set_font("Helvetica", "", 9)
        for r in bad:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                pdf.epw,
                5,
                _pdf_txt(
                    f"  - {r.get('primary')} | {r.get('target_mode')} | L{r.get('lead')}: {r.get('error')}"
                ),
            )

    for r in ok:
        pdf.add_page()
        pair = str(r.get("primary", ""))
        pid = str(r.get("primary_id", pair))
        tm_l = f"{r.get('target_mode', '')}, lead {r.get('lead')}d"
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _pdf_txt(f"{pair}  ({pid})  -  {tm_l}"), ln=1)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(
            0,
            5,
            _pdf_txt(f"Rows used: {r.get('n_rows_used')}  |  Effective end: {r.get('effective_end_date')}"),
            ln=1,
        )
        wp = r.get("oos_span") or {}
        pdf.cell(
            0,
            5,
            _pdf_txt(
                f"WF folds: {wp.get('n_folds')}  |  Train / test per window: {wp.get('train_n')} / {wp.get('test_n')}"
            ),
            ln=1,
        )
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(
            0,
            6,
            _pdf_txt("Best strategies (top 5 by OOS Sharpe, threshold backtest)"),
            ln=1,
        )
        pdf.set_font("Helvetica", "", 9)
        tops = r.get("top_models") or []
        if tops:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 5, _pdf_txt(" > " + " | ".join(tops[:5])))
        pdf.ln(4)

        bt = r.get("backtest_table")
        if bt is None or bt.empty:
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 5, _pdf_txt("No backtest table."), ln=1)
            continue

        pdf.set_font("Helvetica", "B", 8)
        cols = [
            "model",
            "pooled_test_accuracy",
            "sharpe",
            "ann_return",
            "max_drawdown",
            "sortino",
            "hit_rate",
            "pct_traded",
            "n_oos_days",
        ]
        # Total width <= ~190mm for portrait A4 between margins
        wcols = [46, 19, 15, 17, 17, 15, 15, 15, 14]
        hdr = [
            "Model",
            "Pooled acc",
            "Sharpe",
            "AnnRet",
            "MaxDD",
            "Sortino",
            "HitRt",
            "%Traded",
            "n_OOS",
        ]
        pdf.set_x(pdf.l_margin)
        x0 = pdf.l_margin
        for i, h in enumerate(hdr):
            pdf.cell(wcols[i], 5, _pdf_txt(h[:14]), border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        for _, row in bt.iterrows():
            pdf.set_x(x0)
            vals = [
                _pdf_txt(str(row["model"]))[:28],
                _fmt_num(row.get("pooled_test_accuracy"), 4),
                _fmt_num(row.get("sharpe"), 3),
                _fmt_num(row.get("ann_return"), 4),
                _fmt_num(row.get("max_drawdown"), 4),
                _fmt_num(row.get("sortino"), 3),
                _fmt_num(row.get("hit_rate"), 4),
                _fmt_num(row.get("pct_traded"), 4),
                _fmt_num(row.get("n_oos_days"), 0),
            ]
            for i, v in enumerate(vals):
                pdf.cell(wcols[i], 5, _pdf_txt(str(v))[:18], border=1)
            pdf.ln()
            if pdf.get_y() > 270:
                pdf.add_page()
                pdf.set_font("Helvetica", "", 7)

        if "roll_sharpe_mean" in bt.columns:
            top = bt.iloc[0]
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 9)
            bs = top.get("rolling_block_size") or "?"
            pdf.cell(
                0,
                5,
                _pdf_txt(
                    f"Rolling OOS blocks — consecutive rows of stitched hold-out returns "
                    f"(block size requested: {bs}). Stats below: mean / std / min / max across blocks "
                    f"(best Sharpe model: {str(top.get('model', ''))}).",
                ),
                ln=1,
            )
            pdf.set_font("Helvetica", "", 7)
            pdf.cell(
                0,
                4,
                _pdf_txt(
                    f"OOS rows: {top.get('rolling_oos_rows')} | Blocks: {top.get('rolling_n_chunks')} | "
                    f"{top.get('rolling_note') or ''}",
                ),
                ln=1,
            )
            for metric, label in (
                ("sharpe", "Sharpe"),
                ("ann_return", "AnnRet"),
                ("sortino", "Sortino"),
                ("max_drawdown", "MaxDD"),
                ("hit_rate", "HitRt"),
                ("pct_traded", "PctTrd"),
                ("ann_vol", "AnnVol"),
            ):
                km = f"roll_{metric}_mean"
                if km not in bt.columns:
                    continue
                pdf.cell(
                    0,
                    4,
                    _pdf_txt(
                        f"{label}: mean={_fmt_num(top.get(km), 4)} std={_fmt_num(top.get(f'roll_{metric}_std'), 4)} "
                        f"min={_fmt_num(top.get(f'roll_{metric}_min'), 4)} max={_fmt_num(top.get(f'roll_{metric}_max'), 4)}",
                    ),
                    ln=1,
                )
                if pdf.get_y() > 275:
                    pdf.add_page()
                    pdf.set_font("Helvetica", "", 7)

    pdf.output(str(out_path))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    patch_currencies_sys_path()
    try:
        from src.config import FX_PAIR_INSTRUMENTS  # noqa: PLC0415
    except Exception as e:
        raise RuntimeError(
            "Could not import Currencies config. Set FX/Currencies project on PYTHONPATH "
            "or run from FX_NonLinear_Forecast_Direction with Currencies as sibling.",
        ) from e

    p = argparse.ArgumentParser(
        description="Batch FX direction strategy PDF report",
        epilog=(
            "Example (full sweep: binary + ternary, leads 1-10 per pair): "
            "py -m fxnl.batch_strategy_report --end-date 2026-05-01 --n-tail 1000 --window 200 "
            "--train-frac 0.7 --pca 10 --threshold 0.51 --target-modes binary ternary "
            "--lead-min 1 --lead-max 10 --jobs 4"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--end-date", default="2026-05-01", help="Last calendar date included in history tail")
    p.add_argument("--n-tail", type=int, default=1000, help="Use last N observations ending on/before end-date")
    p.add_argument("--window", type=int, default=200, help="Walk-forward window size (bars)")
    p.add_argument("--train-frac", type=float, default=0.7, help="Train fraction inside each window")
    p.add_argument("--pca", type=int, default=10, help="Global PCA components (0 = off)")
    p.add_argument("--threshold", type=float, default=0.51, help="Min max-class probability for trading")
    p.add_argument(
        "--target-modes",
        nargs="+",
        choices=["binary", "ternary"],
        default=["binary", "ternary"],
        metavar="MODE",
        help="Target label mode(s) to run (default: both)",
    )
    p.add_argument("--lead-min", type=int, default=1, help="First forward horizon in days (inclusive)")
    p.add_argument("--lead-max", type=int, default=10, help="Last forward horizon in days (inclusive)")
    p.add_argument(
        "--lead",
        type=int,
        default=None,
        metavar="DAYS",
        help="Run a single lead only (overrides --lead-min / --lead-max)",
    )
    p.add_argument("--no-garch", action="store_true", help="Disable GARCH/vol features")
    p.add_argument("--pairs", nargs="*", default=None, help="Subset of instruments (default: all FX_PAIR_INSTRUMENTS)")
    p.add_argument("--no-cross-pairs", action="store_true", help="Primary only (no other pairs as features)")
    p.add_argument("--jobs", type=int, default=2, help="Parallel threads (I/O bound + sklearn)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output PDF path")
    p.add_argument(
        "--rolling-chunk",
        type=int,
        default=0,
        metavar="N",
        help="If >0, summarize Sharpe/return/etc. across consecutive OOS blocks of N stitched rows (mean/std/min/max)",
    )
    args = p.parse_args(argv)

    pairs = args.pairs or list(FX_PAIR_INSTRUMENTS)
    if args.lead is not None:
        leads_list = [int(args.lead)]
    else:
        lo, hi = int(args.lead_min), int(args.lead_max)
        if lo > hi:
            raise SystemExit("--lead-min must be <= --lead-max")
        leads_list = list(range(lo, hi + 1))
    target_modes = list(args.target_modes)
    out_pdf = args.output or (
        Path(__file__).resolve().parents[1] / f"strategy_report_{args.end_date.replace('-', '')}.pdf"
    )

    def ctx_for(primary: str) -> list[str]:
        if args.no_cross_pairs:
            return [primary]
        return [primary, *[x for x in FX_PAIR_INSTRUMENTS if x != primary]]

    if len(leads_list) == 1:
        leads_desc = str(leads_list[0])
    else:
        leads_desc = f"{min(leads_list)}–{max(leads_list)} ({len(leads_list)} values)"
    try:
        from fxnl.data_repo import data_repo_root, use_repo_data  # noqa: PLC0415

        _repo = data_repo_root()
        _panel = (
            f"ETF fx_data_collect repo ({_repo})"
            if use_repo_data() and _repo is not None
            else "Currencies cache (set FXNL_USE_REPO_DATA=1 and FXNL_DATA_REPO if repo missing)"
        )
    except Exception:
        _panel = "see FXNL_* env"

    params_text = {
        "End date (cap)": args.end_date,
        "Last N observations": args.n_tail,
        "WF window (bars)": args.window,
        "Train / test split": f"{args.train_frac:.0%} / {1 - args.train_frac:.0%}",
        "PCA components": args.pca if args.pca > 0 else "off",
        "Probability threshold": f"{args.threshold:.0%}",
        "Target modes": ", ".join(target_modes),
        "Leads (days)": leads_desc,
        "Pairs": ", ".join(pairs),
        "Cross-pair features": "no" if args.no_cross_pairs else "yes (all others merged)",
        "GARCH features": "no" if args.no_garch else "yes",
        "Feature / fund / tech panel": _panel,
        "Rolling OOS block size (rows)": int(args.rolling_chunk) if args.rolling_chunk > 0 else "off",
    }

    results: list[dict[str, Any]] = []
    combos: list[tuple[str, str, int]] = [
        (primary, tm, ld) for primary in pairs for tm in target_modes for ld in leads_list
    ]

    rc = int(args.rolling_chunk) if args.rolling_chunk > 0 else None

    def job_combo(c: tuple[str, str, int]) -> dict[str, Any]:
        primary, tm, ld = c
        return run_one_pair(
            primary,
            context_instruments=ctx_for(primary),
            end_date=args.end_date,
            n_tail=int(args.n_tail),
            window_size=int(args.window),
            train_frac=float(args.train_frac),
            pca_components=int(args.pca),
            prob_threshold=float(args.threshold),
            lead=int(ld),
            target_mode=tm,
            include_garch=not args.no_garch,
            rolling_chunk_size=rc,
        )

    pair_order = {ins: i for i, ins in enumerate(pairs)}
    mode_order = {"binary": 0, "ternary": 1}

    def _sort_result(r: dict[str, Any]) -> tuple[int, int, int, str]:
        p = str(r.get("primary") or "")
        tm = str(r.get("target_mode") or "")
        ld = int(r.get("lead") or 0)
        return (pair_order.get(p, 999), mode_order.get(tm, 9), ld, p)

    j = max(1, int(args.jobs))
    if j == 1:
        for c in combos:
            results.append(job_combo(c))
    else:
        with ThreadPoolExecutor(max_workers=j) as ex:
            futs = {ex.submit(job_combo, c): c for c in combos}
            for fut in as_completed(futs):
                results.append(fut.result())
    results.sort(key=_sort_result)

    build_pdf(results, out_pdf, params=params_text)
    csv_path = out_pdf.with_suffix(".csv")
    # flatten leaderboard
    all_rows: list[pd.DataFrame] = []
    for r in results:
        if r.get("error") or r.get("backtest_table") is None:
            continue
        bt = r["backtest_table"].copy()
        bt.insert(0, "lead", r.get("lead"))
        bt.insert(0, "target_mode", r.get("target_mode"))
        bt.insert(0, "primary", r.get("primary"))
        all_rows.append(bt)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(csv_path, index=False)
        logging.info("Wrote %s and %s", out_pdf, csv_path)
    else:
        logging.info("Wrote %s (no CSV — no successful runs)", out_pdf)


if __name__ == "__main__":
    main()
