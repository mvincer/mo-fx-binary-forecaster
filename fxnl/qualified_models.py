"""Persist qualified binary FX models from the rolling-batch source workbook.

Endorsement gate (binary FX models only):
  * ``error`` column empty
  * ``target_mode == "binary"``
  * ``mean_seq_oos_acc >= --min-accuracy``  (default 0.60)

The output ``qualified_models.json`` (+ sibling ``.csv``) is consumed by
``fxnl.daily_binary_forecast_report`` so the daily / dashboard pipeline never
spends compute on combos that didn't clear the bar — and the endorsement
survives even if the workbook is later regenerated with different combos.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_qualified_models(
    *,
    source_xlsx: Path,
    out_json: Path,
    min_accuracy: float = 0.60,
    strong_accuracy: float = 0.65,
) -> dict[str, Any]:
    if not source_xlsx.exists():
        raise FileNotFoundError(f"Source workbook not found: {source_xlsx}")

    run_summary = pd.read_excel(source_xlsx, sheet_name="run_summary")
    all_batches = pd.read_excel(source_xlsx, sheet_name="all_batches")

    rs = run_summary.copy()
    rs["error"] = rs["error"].fillna("") if "error" in rs.columns else ""
    rs = rs[
        (rs["error"].astype(str).str.len() == 0)
        & (rs["target_mode"].astype(str).str.lower() == "binary")
        & (pd.to_numeric(rs["mean_seq_oos_acc"], errors="coerce") >= float(min_accuracy))
    ].copy()

    latest_batches = (
        all_batches.sort_values(["run_id", "batch_idx"])
        .groupby("run_id", as_index=False)
        .tail(1)[["run_id", "chosen_variant", "chosen_inner_test_acc", "same_model_as_previous_batch"]]
    )
    rs = rs.merge(latest_batches, on="run_id", how="left")

    rows: list[dict[str, Any]] = []
    for _, r in rs.iterrows():
        acc = float(pd.to_numeric(r.get("mean_seq_oos_acc"), errors="coerce"))
        rows.append(
            {
                "run_id": str(r.get("run_id")),
                "primary": str(r.get("primary")),
                "lead_days": int(r.get("lead")),
                "target_return_basis": str(r.get("target_return_basis")),
                "target_mode": "binary",
                "chosen_variant": str(r.get("chosen_variant", "")).strip(),
                "avg_oos_accuracy": acc,
                "is_strong": bool(acc >= float(strong_accuracy)),
                "global_batch_stability_rate": (
                    float(pd.to_numeric(r.get("global_batch_stability_rate"), errors="coerce"))
                    if pd.notna(r.get("global_batch_stability_rate"))
                    else None
                ),
                "latest_inner_test_acc": (
                    float(pd.to_numeric(r.get("chosen_inner_test_acc"), errors="coerce"))
                    if pd.notna(r.get("chosen_inner_test_acc"))
                    else None
                ),
                "same_as_previous_batch": (
                    bool(r.get("same_model_as_previous_batch"))
                    if pd.notna(r.get("same_model_as_previous_batch"))
                    else None
                ),
            },
        )

    rows.sort(key=lambda d: (-(d["avg_oos_accuracy"] or 0.0), d["primary"], d["lead_days"]))

    payload = {
        "source_xlsx": str(source_xlsx.resolve()),
        "min_accuracy": float(min_accuracy),
        "strong_accuracy": float(strong_accuracy),
        "n_qualified": len(rows),
        "n_strong": int(sum(1 for r in rows if r["is_strong"])),
        "qualified": rows,
        "note": (
            "Endorsed binary FX models with avg sequential OOS accuracy >= min_accuracy on the "
            "1000-bar rolling-batch study. Downstream tools should run only these combos."
        ),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_json.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "primary",
                "lead_days",
                "target_return_basis",
                "chosen_variant",
                "avg_oos_accuracy",
                "is_strong",
                "global_batch_stability_rate",
                "latest_inner_test_acc",
                "same_as_previous_batch",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "primary": r["primary"],
                    "lead_days": r["lead_days"],
                    "target_return_basis": r["target_return_basis"],
                    "chosen_variant": r["chosen_variant"],
                    "avg_oos_accuracy": r["avg_oos_accuracy"],
                    "is_strong": r["is_strong"],
                    "global_batch_stability_rate": r["global_batch_stability_rate"],
                    "latest_inner_test_acc": r["latest_inner_test_acc"],
                    "same_as_previous_batch": r["same_as_previous_batch"],
                },
            )
    logger.info("Wrote %s and %s", out_json, csv_path)
    return payload


def load_qualified_models(json_path: Path) -> list[dict[str, Any]] | None:
    """Endorsed model list, or ``None`` when the file is missing/unreadable."""
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read qualified models %s: %s", json_path, e)
        return None
    rows = payload.get("qualified") or []
    return list(rows) if isinstance(rows, list) else None


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build qualified_models.json from the rolling-batch workbook.")
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--source-xlsx", type=Path, default=root / "batch_hold_full_export.xlsx")
    p.add_argument("--out-json", type=Path, default=root / "qualified_models.json")
    p.add_argument("--min-accuracy", type=float, default=0.60)
    p.add_argument("--strong-accuracy", type=float, default=0.65)
    args = p.parse_args(argv)

    payload = build_qualified_models(
        source_xlsx=args.source_xlsx,
        out_json=args.out_json,
        min_accuracy=float(args.min_accuracy),
        strong_accuracy=float(args.strong_accuracy),
    )
    logging.info(
        "Endorsed %d binary FX models (%d strong) at >= %.2f avg OOS acc.",
        payload["n_qualified"],
        payload["n_strong"],
        payload["min_accuracy"],
    )


if __name__ == "__main__":
    main()
