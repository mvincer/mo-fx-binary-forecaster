"""
Full sweep: batch rolling tune (25-step hold, 200/50 inner split, sticky champion) × pairs × leads × targets × bases.

Writes:
  - Excel (.xlsx) with sheets: run_summary, all_batches, all_steps, model_stability_global
  - PDF summary

Run (from FX_NonLinear_Forecast_Direction):

  py -3 -m fxnl.batch_hold_export_sweep --out-xlsx results.xlsx --out-pdf results.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fxnl.data_panel import prepare_direction_frame
from fxnl.paths import patch_currencies_sys_path
from fxnl.rolling_tune_batch import rolling_tune_batch_holdout_oos

logger = logging.getLogger(__name__)


def _trim_last_n(base: dict[str, Any], n: int) -> dict[str, Any]:
    if n <= 0 or base.get("error"):
        return base
    if len(base["X"]) <= n:
        return base
    out = dict(base)
    out["X"] = base["X"].iloc[-n:]
    out["y"] = base["y"].iloc[-n:]
    out["forward_r"] = base["forward_r"].iloc[-n:]
    out["dates"] = pd.DatetimeIndex(pd.to_datetime(base["dates"]))[-n:]
    if base.get("hi_exc") is not None:
        out["hi_exc"] = base["hi_exc"].iloc[-n:]
    if base.get("lo_exc") is not None:
        out["lo_exc"] = base["lo_exc"].iloc[-n:]
    out["n_total"] = len(out["X"])
    return out


def _context_all(primary: str, all_pairs: list[str]) -> list[str]:
    return [primary, *[x for x in all_pairs if x != primary]]


def _pdf_txt(s: Any) -> str:
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


def _pdf_safe_multiline(pdf: Any, text: str, *, h: float = 5.0) -> None:
    """Robust wrapper to avoid fpdf line-break crashes on long unbreakable tokens."""
    txt = _pdf_txt(text).replace("\t", " ")
    # Hard-wrap very long tokens (paths, reprs) before multi_cell.
    for token in txt.split(" "):
        if len(token) > 80:
            for i in range(0, len(token), 80):
                pdf.multi_cell(pdf.epw, h, token[i : i + 80])
        else:
            pdf.multi_cell(pdf.epw, h, token)


def build_pdf_summary(
    run_summary: pd.DataFrame,
    stability_wide: pd.DataFrame,
    out_path: Path,
    *,
    params: dict[str, Any],
) -> None:
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError as e:
        raise RuntimeError("pip install fpdf2") from e

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _pdf_txt("FX batch rolling tune — summary"), ln=1)
    pdf.set_font("Helvetica", "", 9)
    for k, v in params.items():
        _pdf_safe_multiline(pdf, f"{k}: {v}", h=5)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, _pdf_txt("Per-run headline (mean seq OOS acc, stability)"), ln=1)
    pdf.set_font("Helvetica", "", 7)
    if run_summary is not None and not run_summary.empty:
        cols = [c for c in run_summary.columns if c in run_summary.columns][:12]
        for _, row in run_summary.head(80).iterrows():
            line = " | ".join(f"{c}={row.get(c, '')}" for c in cols)
            _pdf_safe_multiline(pdf, line[:240], h=4)
            if pdf.get_y() > 270:
                pdf.add_page()
                pdf.set_font("Helvetica", "", 7)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, _pdf_txt("Aggregated variant stability (retention after self)"), ln=1)
    pdf.set_font("Helvetica", "", 7)
    if stability_wide is not None and not stability_wide.empty:
        for _, row in stability_wide.head(60).iterrows():
            _pdf_safe_multiline(pdf, str(dict(row))[:300], h=4)
            if pdf.get_y() > 270:
                pdf.add_page()
                pdf.set_font("Helvetica", "", 7)

    pdf.output(str(out_path))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    patch_currencies_sys_path()
    try:
        from src.config import FX_PAIR_INSTRUMENTS  # noqa: PLC0415
    except Exception as e:
        logger.error("Need Currencies FX_PAIR_INSTRUMENTS: %s", e)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Batch hold sweep → Excel + PDF")
    ap.add_argument("--out-xlsx", type=Path, default=Path("batch_hold_export.xlsx"))
    ap.add_argument("--out-pdf", type=Path, default=Path("batch_hold_export.pdf"))
    ap.add_argument("--lookback-bars", type=int, default=1000)
    ap.add_argument("--lead-min", type=int, default=1)
    ap.add_argument("--lead-max", type=int, default=10)
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--pca", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--pdf-only-from-xlsx",
        type=Path,
        default=None,
        help="Skip sweep; build PDF from existing workbook (sheet run_summary + model_stability_global).",
    )
    args = ap.parse_args(argv)

    if args.pdf_only_from_xlsx is not None:
        xlsx_in = args.pdf_only_from_xlsx
        if not xlsx_in.is_file():
            raise FileNotFoundError(f"Workbook not found: {xlsx_in}")
        run_summary = pd.read_excel(xlsx_in, sheet_name="run_summary")
        try:
            agg_stab = pd.read_excel(xlsx_in, sheet_name="model_stability_global")
        except Exception:
            agg_stab = pd.DataFrame()
        build_pdf_summary(
            run_summary,
            agg_stab,
            args.out_pdf,
            params={
                "source_workbook": str(xlsx_in),
                "mode": "pdf-only",
                "rows_run_summary": len(run_summary),
                "rows_model_stability_global": len(agg_stab),
            },
        )
        logging.info("Wrote PDF %s", args.out_pdf.resolve())
        return

    pairs = list(args.pairs) if args.pairs else list(FX_PAIR_INSTRUMENTS)
    leads = list(range(int(args.lead_min), int(args.lead_max) + 1))
    targets = ["binary", "ternary"]
    bases = ["close", "high_low"]

    combos = [(p, tm, tb, ld) for p in pairs for tm in targets for tb in bases for ld in leads]
    logging.info("Total sweep combos: %s", len(combos))

    if args.dry_run:
        print(f"Combos: {len(combos)} -> {args.out_xlsx} , {args.out_pdf}")
        return

    run_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    step_chunks: list[pd.DataFrame] = []
    stab_chunks: list[pd.DataFrame] = []

    t0_all = time.perf_counter()
    for ci, (primary, tm, tb, ld) in enumerate(combos):
        ctx = _context_all(primary, list(FX_PAIR_INSTRUMENTS))
        base = prepare_direction_frame(
            primary,
            context_instruments=ctx,
            period="max",
            study_bars=None,
            lead=int(ld),
            target_mode=str(tm),
            exclude_all_close=True,
            ternary_band=0.005,
            include_garch_vol=True,
            target_return_basis=str(tb),
        )
        rid = f"{primary}|{tm}|{tb}|{ld}"
        if base.get("error"):
            run_rows.append(
                {
                    "run_id": rid,
                    "primary": primary,
                    "target_mode": tm,
                    "target_return_basis": tb,
                    "lead": ld,
                    "error": str(base["error"]),
                    "mean_seq_oos_acc": None,
                    "n_batches": 0,
                    "global_batch_stability_rate": None,
                },
            )
            continue

        base = _trim_last_n(base, int(args.lookback_bars))
        out = rolling_tune_batch_holdout_oos(
            base["X"],
            base["y"],
            base["forward_r"],
            history_bars=int(args.lookback_bars),
            fit_window_bars=250,
            inner_train_n=200,
            inner_te_n=50,
            batch_oos_size=25,
            expanding_pool=True,
            history_cap_bars=None,
            family_filter="faster",
            sticky_batch_champion=True,
            random_state=42,
            tune_n_jobs=-1,
            global_pca_components=int(args.pca) if int(args.pca) > 0 else None,
            max_batches=None,
        )

        if out.get("error"):
            run_rows.append(
                {
                    "run_id": rid,
                    "primary": primary,
                    "target_mode": tm,
                    "target_return_basis": tb,
                    "lead": ld,
                    "error": str(out["error"]),
                    "mean_seq_oos_acc": None,
                    "n_batches": 0,
                    "global_batch_stability_rate": None,
                },
            )
            continue

        ps = out.get("per_step_predictions")
        bs = out.get("batch_summary")
        st = out.get("stability") or {}
        pm = out.get("per_model_stability")

        macc = float(ps["correct"].mean()) if ps is not None and not ps.empty else float("nan")

        run_rows.append(
            {
                "run_id": rid,
                "primary": primary,
                "target_mode": tm,
                "target_return_basis": tb,
                "lead": ld,
                "error": "",
                "mean_seq_oos_acc": macc,
                "n_batches": st.get("n_batches", 0),
                "global_batch_stability_rate": st.get("global_batch_stability_rate"),
                "n_panel_rows": len(base["X"]),
            },
        )

        if bs is not None and not bs.empty:
            bbc = bs.copy()
            bbc.insert(0, "run_id", rid)
            bbc.insert(1, "primary", primary)
            bbc.insert(2, "target_mode", tm)
            bbc.insert(3, "target_return_basis", tb)
            bbc.insert(4, "lead", ld)
            batch_rows.extend(bbc.to_dict("records"))

        if ps is not None and not ps.empty:
            p2 = ps.copy()
            p2.insert(0, "run_id", rid)
            p2.insert(1, "primary", primary)
            step_chunks.append(p2)

        if pm is not None and not pm.empty:
            pm2 = pm.copy()
            pm2.insert(0, "run_id", rid)
            pm2.insert(1, "primary", primary)
            stab_chunks.append(pm2)

        if (ci + 1) % 5 == 0:
            logging.info("Finished %s / %s runs (%.1fs)", ci + 1, len(combos), time.perf_counter() - t0_all)

    run_summary = pd.DataFrame(run_rows)
    all_batches = pd.DataFrame(batch_rows)
    all_steps = pd.concat(step_chunks, ignore_index=True) if step_chunks else pd.DataFrame()
    model_stab = pd.concat(stab_chunks, ignore_index=True) if stab_chunks else pd.DataFrame()

    # Global aggregation across runs per variant
    agg_stab = pd.DataFrame()
    if not model_stab.empty:
        agg_stab = (
            model_stab.groupby("variant", dropna=False)
            .agg(
                batches_as_champion=("batches_as_champion", "sum"),
                consecutive_pairs=("consecutive_same_model_pairs", "sum"),
                mean_retention=("retention_after_self_rate", "mean"),
            )
            .reset_index()
            .sort_values("batches_as_champion", ascending=False)
        )

    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        run_summary.to_excel(writer, sheet_name="run_summary", index=False)
        all_batches.to_excel(writer, sheet_name="all_batches", index=False)
        if not all_steps.empty:
            all_steps.to_excel(writer, sheet_name="all_steps", index=False)
        model_stab.to_excel(writer, sheet_name="per_run_model_stability", index=False)
        agg_stab.to_excel(writer, sheet_name="model_stability_global", index=False)

    logging.info("Wrote Excel %s", args.out_xlsx.resolve())

    build_pdf_summary(
        run_summary,
        agg_stab,
        args.out_pdf,
        params={
            "lookback_bars": args.lookback_bars,
            "inner_split": "200 train / 50 test (250 tune window)",
            "batch_oos_size": 25,
            "pca": args.pca,
            "leads": f"{args.lead_min}-{args.lead_max}",
            "pairs": len(pairs),
            "combos": len(combos),
            "elapsed_sec": round(time.perf_counter() - t0_all, 1),
        },
    )
    logging.info("Wrote PDF %s", args.out_pdf.resolve())


if __name__ == "__main__":
    main()
