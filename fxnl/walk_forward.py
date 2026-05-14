"""Rolling walk-forward: 80% train / 20% test inside each window; stitch OOS predictions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from fxnl.metrics_eval import eval_split
from fxnl.model_fit_core import fit_models_core, preprocess_direction_xy

logger = logging.getLogger(__name__)


def _row_calendar(X: pd.DataFrame) -> pd.DatetimeIndex:
    """One timestamp per row from the frame index (avoids reindex bugs with stored ``dates`` series)."""
    ix = X.index
    if isinstance(ix, pd.DatetimeIndex):
        return ix
    try:
        return pd.DatetimeIndex(np.asarray(ix, dtype="datetime64[ns]"))
    except (ValueError, TypeError):
        return pd.DatetimeIndex(pd.to_datetime(np.asarray(ix, dtype=object)))


def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    forward_r: pd.Series,
    *,
    window_size: int,
    train_frac: float = 0.8,
    step: int | None = None,
    lookback_bars: int | None = None,
    random_state: int = 42,
    global_pca_components: int | None = None,
    skip_lstm: bool = True,
    pair: str = "",
    lead: int = 1,
    target_mode: str = "binary",
    models_include: list[str] | None = None,
) -> dict[str, Any]:
    """
    Slide a window of ``window_size`` rows; within each window use first ``train_frac`` for training.

    Default ``step`` = test segment length (``window_size * (1 - train_frac)``) so test folds abut without overlap.

    Row calendar labels OOS segments using ``X.index`` only (must be parseable as datetimes).
    """
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac must be in (0, 1)")
    if window_size < 50:
        raise ValueError("window_size should be at least 50 for stable fits")

    X = X.copy().sort_index()
    y = y.reindex(X.index)
    forward_r = forward_r.reindex(X.index)

    if lookback_bars is not None and int(lookback_bars) > 0:
        lb = min(len(X), int(lookback_bars))
        X = X.iloc[-lb:]
        y = y.loc[X.index]
        forward_r = forward_r.loc[X.index]

    # Calendar: always ``X.index`` (do not pass a separate ``dates`` array — reindex was error-prone).
    row_dates = _row_calendar(X)

    test_n = int(round(window_size * (1.0 - train_frac)))
    train_n = window_size - test_n
    if test_n < 1 or train_n < 20:
        return {"error": f"bad_split train={train_n} test={test_n} for window={window_size}"}

    if step is None or int(step) <= 0:
        step = test_n

    n = len(X)
    if n < window_size:
        return {"error": f"too_few_rows:{n} for window {window_size}"}

    all_fold_rows: list[pd.DataFrame] = []
    oos_parts: list[dict[str, Any]] = []

    for start in range(0, n - window_size + 1, int(step)):
        end = start + window_size
        X_w = X.iloc[start:end]
        y_w = y.iloc[start:end].to_numpy()
        r_w = forward_r.iloc[start:end].to_numpy()
        dates_w = row_dates[start:end]

        X_tr, X_te = X_w.iloc[:train_n], X_w.iloc[train_n:]
        y_tr, y_te = y_w[:train_n], y_w[train_n:]
        r_te = r_w[train_n:]
        d_te = dates_w[train_n:]

        try:
            pre = preprocess_direction_xy(
                X_tr,
                X_te,
                y_tr,
                y_te,
                random_state=random_state,
                global_pca_components=global_pca_components,
            )
            res_df, preds, _acc_map = fit_models_core(
                pre,
                pair=pair,
                lead=lead,
                target_mode=target_mode,
                n_train=len(X_tr),
                n_test=len(X_te),
                random_state=random_state,
                skip_lstm=skip_lstm,
                models_include=models_include,
            )
        except Exception as e:
            logger.exception("walk-forward fold failed start=%s", start)
            continue

        if res_df.empty:
            continue

        res_df = res_df.copy()
        res_df.insert(0, "wf_window_start", start)
        res_df.insert(1, "wf_window_end", end - 1)
        all_fold_rows.append(res_df)

        block = {
            "start": start,
            "end": end,
            "dates_te": d_te,
            "y_te": y_te,
            "forward_r_te": r_te,
            "preds": preds,
        }
        oos_parts.append(block)

    if not all_fold_rows:
        return {"error": "no_successful_folds", "fold_results": pd.DataFrame(), "oos": []}

    fold_results = pd.concat(all_fold_rows, axis=0, ignore_index=True)

    # Summary: mean test metrics by model across folds
    metric_cols = [c for c in fold_results.columns if c.startswith("test_")]
    summary = fold_results.groupby("model", dropna=False)[metric_cols].mean().reset_index()

    # Pooled OOS: concatenate all test segments (non-overlapping if step==test_n)
    pooled_summary_rows: list[dict[str, Any]] = []
    for mname in fold_results["model"].unique():
        ys: list[np.ndarray] = []
        Ps: list[np.ndarray] = []
        for block in oos_parts:
            pr = block["preds"].get(mname)
            if pr is None or pr.get("proba_te") is None:
                continue
            pt = pr["proba_te"]
            ys.append(block["y_te"])
            Ps.append(pt)
        if not ys:
            continue
        y_cat = np.concatenate(ys)
        P = np.vstack(Ps)
        if len(y_cat) != len(P):
            continue
        pred = np.argmax(P, axis=1)
        te_m = eval_split(y_cat, pred, P, n_classes=int(P.shape[1]))
        pooled_summary_rows.append(
            {
                "model": mname,
                "pooled_n": len(y_cat),
                "pooled_test_accuracy": te_m["accuracy"],
                "pooled_test_balanced_acc": te_m["balanced_accuracy"],
                "pooled_test_log_loss": te_m.get("log_loss", float("nan")),
                "pooled_test_f1_macro": te_m.get("f1_macro", float("nan")),
            },
        )

    pooled_df = pd.DataFrame(pooled_summary_rows)

    return {
        "fold_results": fold_results,
        "summary_by_fold": summary,
        "pooled_oos_metrics": pooled_df,
        "oos_parts": oos_parts,
        "window_params": {
            "window_size": window_size,
            "train_frac": train_frac,
            "train_n": train_n,
            "test_n": test_n,
            "step": int(step),
            "n_folds": len(oos_parts),
            "lookback_bars": lookback_bars,
        },
    }
