"""
Offline sweep: walk-forward (80/20 inside window), rolling tune + sequential OOS, hold-out (50/50).

Mirrors the Streamlit app pipelines with fixed knobs for batch runs (PCA k=5, GARCH on, cross-pair features).

Example (dry-run counts only):

  cd FX_NonLinear_Forecast_Direction
  py -m fxnl.batch_eval_sweep --dry-run

Full sweep is **very large** (pairs × leads × 2 targets × 2 bases × 3 modes). Use filters:

  py -m fxnl.batch_eval_sweep --pairs EUR/USD --lead-min 1 --lead-max 2 --eval-modes wf holdout --out sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fxnl.data_panel import build_direction_dataset, prepare_direction_frame
from fxnl.models_runner import run_direction_models_with_preds
from fxnl.paths import patch_currencies_sys_path
from fxnl.rolling_tune_oos import rolling_tune_sequential_oos
from fxnl.walk_forward import walk_forward_evaluate

logger = logging.getLogger(__name__)

# --- User-requested defaults (override via CLI) ---
DEFAULT_LOOKBACK_BARS = 1000
DEFAULT_WF_WINDOW = 250
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_WF_STEP = 20
DEFAULT_PCA = 5
DEFAULT_ROLL_HISTORY = 1000
DEFAULT_ROLL_FIT_WINDOW = 250
DEFAULT_ROLL_STEP = 20
DEFAULT_FAMILY = "faster"


def _context_all_pairs(primary: str, all_pairs: list[str]) -> list[str]:
    return [primary, *[x for x in all_pairs if x != primary]]


def _trim_panel(base: dict[str, Any], lookback: int) -> dict[str, Any]:
    if lookback <= 0 or base.get("error"):
        return base
    lb = int(lookback)
    if len(base["X"]) <= lb:
        return base
    out = dict(base)
    out["X"] = base["X"].iloc[-lb:]
    out["y"] = base["y"].iloc[-lb:]
    out["forward_r"] = base["forward_r"].iloc[-lb:]
    out["dates"] = pd.DatetimeIndex(pd.to_datetime(base["dates"]))[-lb:]
    if base.get("hi_exc") is not None:
        out["hi_exc"] = base["hi_exc"].iloc[-lb:]
    if base.get("lo_exc") is not None:
        out["lo_exc"] = base["lo_exc"].iloc[-lb:]
    out["n_total"] = len(out["X"])
    return out


def _basis_label(s: str) -> str:
    return "high_low" if str(s).strip().lower() in ("high_low", "hl", "path") else "close"


def run_walk_forward_pack(
    primary: str,
    ctx: list[str],
    *,
    lead: int,
    target_mode: str,
    target_return_basis: str,
    lookback_bars: int,
    window_size: int,
    train_frac: float,
    step: int,
    pca_k: int,
    include_garch: bool,
) -> dict[str, Any]:
    base = prepare_direction_frame(
        primary,
        context_instruments=ctx,
        period="max",
        study_bars=None,
        lead=int(lead),
        target_mode=str(target_mode),
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=include_garch,
        target_return_basis=str(target_return_basis),
    )
    if base.get("error"):
        return {"error": str(base["error"])}
    base = _trim_panel(base, lookback_bars)
    if len(base["X"]) < window_size:
        return {"error": f"too_few_rows_after_trim:{len(base['X'])} need {window_size}"}

    wf = walk_forward_evaluate(
        base["X"],
        base["y"],
        base["forward_r"],
        window_size=int(window_size),
        train_frac=float(train_frac),
        step=int(step),
        lookback_bars=None,
        random_state=42,
        global_pca_components=int(pca_k) if pca_k > 0 else None,
        skip_lstm=True,
        pair=str(base.get("primary_id", primary)),
        lead=int(lead),
        target_mode=str(target_mode),
    )
    if wf.get("error"):
        return {"error": str(wf["error"]), "wf": wf}

    pooled = wf.get("pooled_oos_metrics")
    best_acc = float("nan")
    best_model = ""
    if pooled is not None and not pooled.empty and "pooled_test_accuracy" in pooled.columns:
        j = int(pooled["pooled_test_accuracy"].astype(float).idxmax())
        best_acc = float(pooled.loc[j, "pooled_test_accuracy"])
        best_model = str(pooled.loc[j, "model"])
    wp = wf.get("window_params") or {}
    return {
        "error": None,
        "metric_name": "best_pooled_test_accuracy",
        "metric_value": best_acc,
        "best_model": best_model,
        "n_folds": wp.get("n_folds"),
        "train_n": wp.get("train_n"),
        "test_n": wp.get("test_n"),
    }


def run_holdout_pack(
    primary: str,
    ctx: list[str],
    *,
    lead: int,
    target_mode: str,
    target_return_basis: str,
    study_bars: int,
    pca_k: int,
    include_garch: bool,
) -> dict[str, Any]:
    dset = build_direction_dataset(
        primary,
        context_instruments=ctx,
        period="max",
        study_bars=int(study_bars) if study_bars > 0 else None,
        lead=int(lead),
        target_mode=str(target_mode),
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=include_garch,
        target_return_basis=str(target_return_basis),
    )
    if dset.get("error"):
        return {"error": str(dset["error"])}
    res, _preds = run_direction_models_with_preds(
        dset,
        random_state=42,
        global_pca_components=int(pca_k) if pca_k > 0 else None,
        skip_lstm=True,
    )
    if res.empty or "error" in res.columns:
        er = res["error"].iloc[0] if not res.empty and "error" in res.columns else "empty_results"
        return {"error": str(er)}
    if "test_accuracy" not in res.columns:
        return {"error": "no_test_accuracy_column"}
    j = int(res["test_accuracy"].astype(float).idxmax())
    return {
        "error": None,
        "metric_name": "best_test_accuracy",
        "metric_value": float(res.loc[j, "test_accuracy"]),
        "best_model": str(res.loc[j, "model"]),
        "n_train": int(dset.get("n_train", 0)),
        "n_test": int(dset.get("n_test", 0)),
    }


def run_rolling_pack(
    primary: str,
    ctx: list[str],
    *,
    lead: int,
    target_mode: str,
    target_return_basis: str,
    lookback_bars: int,
    history_pool: int,
    fit_window: int,
    train_frac_inner: float,
    forecast_step: int,
    pca_k: int,
    include_garch: bool,
    expanding_pool: bool,
    inner_test_frac: float,
    family: str,
    max_steps: int,
    tune_n_jobs: int,
) -> dict[str, Any]:
    base = prepare_direction_frame(
        primary,
        context_instruments=ctx,
        period="max",
        study_bars=None,
        lead=int(lead),
        target_mode=str(target_mode),
        exclude_all_close=True,
        ternary_band=0.005,
        include_garch_vol=include_garch,
        target_return_basis=str(target_return_basis),
    )
    if base.get("error"):
        return {"error": str(base["error"])}
    base = _trim_panel(base, lookback_bars)
    n = len(base["X"])
    if n < fit_window + 2:
        return {"error": f"too_few_rows:{n} need > fit_window {fit_window}"}

    ms = None if int(max_steps) <= 0 else int(max_steps)
    rt = rolling_tune_sequential_oos(
        base["X"],
        base["y"],
        base["forward_r"],
        history_bars=int(history_pool),
        fit_window_bars=int(fit_window),
        train_frac=float(train_frac_inner),
        step=int(forecast_step),
        family_filter=str(family).strip().lower(),
        max_steps=ms,
        random_state=42,
        expanding_pool=bool(expanding_pool),
        history_cap_bars=None,
        inner_test_frac=float(inner_test_frac),
        sticky_champion=False,
        rolling_metrics_chunk=0,
        tune_n_jobs=int(tune_n_jobs),
        global_pca_components=int(pca_k) if pca_k > 0 else None,
    )
    if rt.get("error"):
        return {"error": str(rt.get("error")), "hint": rt.get("hint")}

    summ = rt.get("summary_by_model")
    best_acc = float("nan")
    best_model = ""
    if summ is not None and not summ.empty and "seq_oos_accuracy" in summ.columns:
        summ = summ.sort_values("seq_oos_accuracy", ascending=False)
        best_acc = float(summ.iloc[0]["seq_oos_accuracy"])
        best_model = str(summ.iloc[0]["model"])
    pp = rt.get("params") or {}
    return {
        "error": None,
        "metric_name": "best_seq_oos_accuracy",
        "metric_value": best_acc,
        "best_model": best_model,
        "n_forecast_origins": pp.get("n_forecast_origins"),
        "n_rows_panel": n,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    patch_currencies_sys_path()
    try:
        from src.config import FX_PAIR_INSTRUMENTS  # noqa: PLC0415
    except Exception as e:
        logging.error(
            "Need Currencies `src.config.FX_PAIR_INSTRUMENTS` on PYTHONPATH (sibling project). %s",
            e,
        )
        sys.exit(1)

    ap = argparse.ArgumentParser(
        description="Batch walk-forward, rolling tune, hold-out across FX pairs / leads / targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--out", type=Path, default=Path("batch_eval_sweep_results.csv"))
    ap.add_argument("--dry-run", action="store_true", help="Print combination counts and exit")
    ap.add_argument(
        "--eval-modes",
        nargs="+",
        choices=["walk_forward", "rolling_tune", "holdout"],
        default=["walk_forward", "rolling_tune", "holdout"],
    )
    ap.add_argument("--pairs", nargs="*", default=None, help="Subset of instruments (default: all)")
    ap.add_argument("--lead-min", type=int, default=1)
    ap.add_argument("--lead-max", type=int, default=10)
    ap.add_argument("--lookback-bars", type=int, default=DEFAULT_LOOKBACK_BARS)
    ap.add_argument("--wf-window", type=int, default=DEFAULT_WF_WINDOW)
    ap.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    ap.add_argument("--wf-step", type=int, default=DEFAULT_WF_STEP)
    ap.add_argument("--pca", type=int, default=DEFAULT_PCA)
    ap.add_argument("--study-bars", type=int, default=DEFAULT_LOOKBACK_BARS, help="Hold-out panel tail length")
    ap.add_argument("--roll-history", type=int, default=DEFAULT_ROLL_HISTORY)
    ap.add_argument("--roll-fit-window", type=int, default=DEFAULT_ROLL_FIT_WINDOW)
    ap.add_argument("--roll-step", type=int, default=DEFAULT_ROLL_STEP)
    ap.add_argument("--roll-inner-test-frac", type=float, default=0.0)
    ap.add_argument("--roll-family", type=str, default=DEFAULT_FAMILY)
    ap.add_argument("--roll-max-steps", type=int, default=0, help="0 = all origins (slow)")
    ap.add_argument("--tune-n-jobs", type=int, default=-1)
    ap.add_argument("--no-garch", action="store_true")
    ap.add_argument("--no-expanding-pool", action="store_true", help="Rolling: fixed history pool")
    ap.add_argument("--skip-high-low", action="store_true", help="Only close-to-close target basis")
    ap.add_argument("--jobs", type=int, default=1, help="Thread workers (1 = sequential)")
    args = ap.parse_args(argv)

    pairs = list(args.pairs) if args.pairs else list(FX_PAIR_INSTRUMENTS)
    leads = list(range(int(args.lead_min), int(args.lead_max) + 1))
    targets = ["binary", "ternary"]
    bases = ["close", "high_low"]
    if args.skip_high_low:
        bases = ["close"]

    combos: list[tuple[str, str, str, int, str]] = []
    for p in pairs:
        ctx = _context_all_pairs(p, list(FX_PAIR_INSTRUMENTS))
        for tm in targets:
            for tb in bases:
                for ld in leads:
                    for ev in args.eval_modes:
                        combos.append((p, tm, _basis_label(tb), ld, ev))

    logging.info(
        "Sweep size: %s runs (%s pairs × %s leads × %s targets × %s bases × %s modes)",
        len(combos),
        len(pairs),
        len(leads),
        len(targets),
        len(bases),
        len(args.eval_modes),
    )

    if args.dry_run:
        print(f"Would run {len(combos)} evaluations -> {args.out.resolve()}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ts_wall",
        "primary",
        "target_mode",
        "target_return_basis",
        "lead",
        "eval_mode",
        "error",
        "metric_name",
        "metric_value",
        "best_model",
        "elapsed_sec",
        "extra_json",
    ]
    write_header = not args.out.is_file()
    f_out = args.out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    include_garch = not args.no_garch
    expanding = not args.no_expanding_pool

    def one(job: tuple[str, str, str, int, str]) -> dict[str, Any]:
        primary, tm, tb, ld, ev = job
        ctx = _context_all_pairs(primary, list(FX_PAIR_INSTRUMENTS))
        t0 = time.perf_counter()
        row: dict[str, Any] = {
            "ts_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary": primary,
            "target_mode": tm,
            "target_return_basis": tb,
            "lead": ld,
            "eval_mode": ev,
            "error": "",
            "metric_name": "",
            "metric_value": "",
            "best_model": "",
            "elapsed_sec": 0.0,
            "extra_json": "",
        }
        extra: dict[str, Any] = {}
        try:
            if ev == "walk_forward":
                r = run_walk_forward_pack(
                    primary,
                    ctx,
                    lead=ld,
                    target_mode=tm,
                    target_return_basis=tb,
                    lookback_bars=int(args.lookback_bars),
                    window_size=int(args.wf_window),
                    train_frac=float(args.train_frac),
                    step=int(args.wf_step),
                    pca_k=int(args.pca),
                    include_garch=include_garch,
                )
            elif ev == "holdout":
                r = run_holdout_pack(
                    primary,
                    ctx,
                    lead=ld,
                    target_mode=tm,
                    target_return_basis=tb,
                    study_bars=int(args.study_bars),
                    pca_k=int(args.pca),
                    include_garch=include_garch,
                )
            else:
                r = run_rolling_pack(
                    primary,
                    ctx,
                    lead=ld,
                    target_mode=tm,
                    target_return_basis=tb,
                    lookback_bars=int(args.lookback_bars),
                    history_pool=int(args.roll_history),
                    fit_window=int(args.roll_fit_window),
                    train_frac_inner=float(args.train_frac),
                    forecast_step=int(args.roll_step),
                    pca_k=int(args.pca),
                    include_garch=include_garch,
                    expanding_pool=expanding,
                    inner_test_frac=float(args.roll_inner_test_frac),
                    family=str(args.roll_family),
                    max_steps=int(args.roll_max_steps),
                    tune_n_jobs=int(args.tune_n_jobs),
                )
            el = time.perf_counter() - t0
            if r.get("error"):
                row["error"] = str(r["error"])
                extra = {k: v for k, v in r.items() if k != "error"}
            else:
                row["metric_name"] = str(r.get("metric_name", ""))
                row["metric_value"] = r.get("metric_value", "")
                row["best_model"] = str(r.get("best_model", ""))
                extra = {k: v for k, v in r.items() if k not in ("metric_name", "metric_value", "best_model", "error")}
            row["elapsed_sec"] = round(el, 3)
            row["extra_json"] = json.dumps(extra, default=str)[:8000]
        except Exception as e:
            row["error"] = str(e)
            row["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return row

    j = max(1, int(args.jobs))
    write_lock = threading.Lock()

    def write_row(row: dict[str, Any]) -> None:
        with write_lock:
            writer.writerow(row)
            f_out.flush()

    if j == 1:
        for i, c in enumerate(combos):
            write_row(one(c))
            if (i + 1) % 20 == 0:
                logging.info("Progress %s / %s", i + 1, len(combos))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=j) as ex:
            futs = {ex.submit(one, c): c for c in combos}
            done = 0
            for fut in as_completed(futs):
                write_row(fut.result())
                done += 1
                if done % 20 == 0:
                    logging.info("Progress %s / %s", done, len(combos))

    f_out.close()
    logging.info("Wrote %s", args.out.resolve())


if __name__ == "__main__":
    main()
