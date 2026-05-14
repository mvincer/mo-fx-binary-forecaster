"""
Rolling tune with **batch OOS**: tune once per batch on inner train/test (e.g. 200/50 of 250 bars),
deploy the chosen variant for ``batch_oos_size`` consecutive origins, then re-tune.

Supports **batch-level sticky champion**: keep the previous batch's winning variant if its inner-test
accuracy ≥ the best newly tuned variant's inner-test accuracy.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score

from fxnl.rolling_tune_oos import (
    _grid_variants_for_family,
    _prep_deploy_full_one,
    _prep_train_te,
)

logger = logging.getLogger(__name__)


def _score_variant_tt(
    fit_fn: Any,
    Xtrp: np.ndarray,
    y_tr: np.ndarray,
    Xtep: np.ndarray,
    y_te: np.ndarray,
    display_name: str,
) -> tuple[float, str, Any] | None:
    try:
        _pr, pred_te, _ = fit_fn(Xtrp, y_tr, Xtep)
        return (float(accuracy_score(y_te, pred_te)), display_name, fit_fn)
    except Exception as e:
        logger.debug("score_variant_tt %s: %s", display_name, e)
        return None


def rolling_tune_batch_holdout_oos(
    X: pd.DataFrame,
    y: pd.Series,
    forward_r: pd.Series,
    *,
    history_bars: int,
    fit_window_bars: int = 250,
    inner_train_n: int = 200,
    inner_te_n: int = 50,
    batch_oos_size: int = 25,
    expanding_pool: bool = True,
    history_cap_bars: int | None = None,
    family_filter: str = "faster",
    sticky_batch_champion: bool = True,
    random_state: int = 42,
    tune_n_jobs: int = -1,
    global_pca_components: int | None = 5,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """
    Inner window ``fit_window_bars`` split into ``inner_train_n`` + ``inner_te_n`` (no validation band).
    Once per batch start ``t_batch``, each grid variant is fit on train columns and scored on inner test rows.
    Best test accuracy wins; sticky compares previous champion on the **same** inner test slice.

    Then for ``offset in range(batch_oos_size)``, forecast at ``t = t_batch + offset`` refitting the chosen
    variant on the history pool ending at ``t``.
    """
    tw = int(fit_window_bars)
    itr = int(inner_train_n)
    ite = int(inner_te_n)
    if itr + ite != tw:
        return {"error": f"inner_train_n + inner_te_n must equal fit_window_bars ({itr}+{ite}!={tw})"}
    B = max(1, int(batch_oos_size))
    cap = int(history_cap_bars) if history_cap_bars is not None else int(history_bars)

    if expanding_pool:
        if cap < 80:
            return {"error": "cap too small"}
        if tw > cap:
            return {"error": "fit_window_bars cannot exceed history cap"}
        t0 = tw
    else:
        if int(history_bars) < 80:
            return {"error": "history_bars too small"}
        if tw > int(history_bars):
            return {"error": "fit_window > history_bars"}
        t0 = int(history_bars)

    X = X.copy().sort_index()
    y = y.reindex(X.index)
    forward_r = forward_r.reindex(X.index)
    row_dates = pd.DatetimeIndex(pd.to_datetime(X.index))
    n = len(X)
    X_np = np.asarray(X, dtype=float)
    y_np_full = np.asarray(y, dtype=float)

    if n < t0 + B:
        return {
            "error": "rolling_batch_too_few_rows",
            "n_rows": n,
            "need_at_least": t0 + B,
        }

    batch_starts = list(range(t0, n - B + 1, B))
    if max_batches is not None and int(max_batches) > 0:
        batch_starts = batch_starts[: int(max_batches)]

    if not batch_starts:
        return {"error": "no_batch_starts", "per_step_predictions": pd.DataFrame(), "batch_summary": pd.DataFrame()}

    tnj = int(tune_n_jobs)
    if tnj == -1:
        eff_tune_jobs = max(1, min(16, os.cpu_count() or 4))
    elif tnj <= 0:
        eff_tune_jobs = 1
    else:
        eff_tune_jobs = max(1, min(32, tnj))
    parallel_tune = eff_tune_jobs > 1
    rf_n_jobs = 1 if parallel_tune else -1

    per_step_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    oos_parts: list[dict[str, Any]] = []
    diag: dict[str, int] = {
        "skipped_tune_single_class": 0,
        "skipped_pool_single_class": 0,
        "skipped_tune_all_failed": 0,
        "skipped_deploy_failed": 0,
    }

    champion_display_name: str | None = None

    for batch_idx, t_batch in enumerate(batch_starts):
        prev_champion = champion_display_name

        if expanding_pool:
            start_all_cap = max(0, t_batch - cap)
        else:
            start_all_cap = t_batch - int(history_bars)

        S_tune_X = X_np[t_batch - tw : t_batch]
        S_tune_y_raw = y_np_full[t_batch - tw : t_batch]
        if np.unique(S_tune_y_raw.astype(int)).size < 2:
            diag["skipped_tune_single_class"] += 1
            continue

        X_tr_in = S_tune_X[:itr]
        X_te_in = S_tune_X[itr:]
        S_tune_y_map = S_tune_y_raw.astype(int)
        y_tr_in = S_tune_y_map[:itr]
        y_te_in = S_tune_y_map[itr:]

        u_labels = np.sort(np.unique(np.concatenate([S_tune_y_raw.ravel()]).astype(int)))
        to_idx = {int(v): i for i, v in enumerate(u_labels)}
        n_cls = int(len(u_labels))
        if n_cls < 2:
            diag["skipped_tune_single_class"] += 1
            continue

        y_tr_m = np.array([to_idx[int(v)] for v in y_tr_in], dtype=int)
        y_te_m = np.array([to_idx[int(v)] for v in y_te_in], dtype=int)

        Xtrp, Xtep = _prep_train_te(X_tr_in, X_te_in, global_pca_components, random_state)

        grids = _grid_variants_for_family(family_filter, n_cls, rf_n_jobs=rf_n_jobs)

        if parallel_tune:
            raw_scores = Parallel(n_jobs=eff_tune_jobs, batch_size=1)(
                delayed(_score_variant_tt)(fn, Xtrp, y_tr_m, Xtep, y_te_m, dn)
                for dn, _k, fn in grids
            )
            scored = [x for x in raw_scores if x is not None]
        else:
            scored = []
            for dn, _k, fn in grids:
                r = _score_variant_tt(fn, Xtrp, y_tr_m, Xtep, y_te_m, dn)
                if r is not None:
                    scored.append(r)

        if not scored:
            diag["skipped_tune_all_failed"] += 1
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_new_test, dn_new, fn_new = scored[0][0], scored[0][1], scored[0][2]

        use_display = dn_new
        use_fn = fn_new
        use_test = best_new_test
        sticky_used = False

        if sticky_batch_champion and prev_champion:
            ch_fn = None
            for dn_s, _ks, fn_s in grids:
                if dn_s == prev_champion:
                    ch_fn = fn_s
                    break
            if ch_fn is not None:
                try:
                    _pr, pred_te, _ = ch_fn(Xtrp, y_tr_m, Xtep)
                    ch_test = float(accuracy_score(y_te_m, pred_te))
                    if ch_test >= best_new_test - 1e-15:
                        use_display = prev_champion
                        use_fn = ch_fn
                        use_test = ch_test
                        sticky_used = bool(prev_champion != dn_new)
                except Exception as e:
                    logger.debug("sticky batch: %s", e)

        same_as_previous = bool(prev_champion is not None and use_display == prev_champion)

        batch_ok = False

        for offset in range(B):
            t = t_batch + offset
            if t >= n:
                break

            if expanding_pool:
                start_all = max(0, t - cap)
                S_all_X = X_np[start_all:t]
                S_all_y_raw = y_np_full[start_all:t]
            else:
                hb = int(history_bars)
                S_all_X = X_np[t - hb : t]
                S_all_y_raw = y_np_full[t - hb : t]

            X_one = X_np[t : t + 1]
            y_t_raw = float(y_np_full[t])
            r_t = float(forward_r.iloc[t])

            u2 = np.sort(
                np.unique(np.concatenate([S_all_y_raw.ravel(), np.array([y_t_raw])]).astype(int)),
            )
            to2 = {int(v): i for i, v in enumerate(u2)}
            n_cls2 = int(len(u2))
            S_all_y_map = np.array([to2[int(v)] for v in S_all_y_raw], dtype=int)
            y_t_m = int(to2[int(y_t_raw)])

            if n_cls2 < 2:
                diag["skipped_pool_single_class"] += 1
                continue

            prefix = use_display.split()[0]
            label_name = f"{prefix} (batch-tuned)"

            try:
                Xfull, Xpred = _prep_deploy_full_one(
                    S_all_X, X_one, global_pca_components, random_state
                )
                proba_o, pred_o, _ = use_fn(Xfull, S_all_y_map, Xpred)
                proba_o = np.asarray(proba_o, dtype=float)
                if proba_o.ndim == 1:
                    proba_o = proba_o.reshape(1, -1)
                pred_mapped = int(np.argmax(proba_o, axis=1)[0])
                correct = int(pred_mapped == y_t_m)

                per_step_rows.append(
                    {
                        "batch_idx": batch_idx,
                        "t_batch_start": t_batch,
                        "offset_in_batch": offset,
                        "forecast_idx": t,
                        "date": row_dates[t],
                        "model": label_name,
                        "chosen_variant": use_display,
                        "y_true": y_t_m,
                        "y_true_raw": int(y_t_raw),
                        "pred_class": pred_mapped,
                        "correct": correct,
                        "inner_test_accuracy_at_tune": use_test,
                        "forward_r": r_t,
                        "sticky_batch_champion": sticky_used,
                        "same_model_as_previous_batch": same_as_previous,
                    },
                )
                batch_ok = True
            except Exception as e:
                logger.debug("deploy t=%s: %s", t, e)
                diag["skipped_deploy_failed"] += 1
                continue

        if batch_ok:
            champion_display_name = use_display
            batch_rows.append(
                {
                    "batch_idx": batch_idx,
                    "t_batch_start": t_batch,
                    "t_batch_end_forecast": min(t_batch + B - 1, n - 1),
                    "chosen_variant": use_display,
                    "best_new_inner_test_acc": best_new_test,
                    "chosen_inner_test_acc": use_test,
                    "sticky_used_new_beats": sticky_used,
                    "same_model_as_previous_batch": same_as_previous,
                    "previous_batch_champion": prev_champion or "",
                },
            )
    per_step_df = pd.DataFrame(per_step_rows)
    batch_df = pd.DataFrame(batch_rows)

    stability_global = float("nan")
    per_model_stability: list[dict[str, Any]] = []
    if len(batch_df) >= 2:
        sm = batch_df["same_model_as_previous_batch"].astype(bool)
        stability_global = float(sm.sum() / max(1, len(batch_df) - 1))

    if not batch_df.empty:
        ch = batch_df["chosen_variant"].astype(str).values
        for m in sorted(set(ch)):
            n_pick = int((ch == m).sum())
            cont = sum(1 for i in range(1, len(ch)) if ch[i] == m and ch[i - 1] == m)
            prev_m = batch_df["chosen_variant"].shift(1).astype(str) == m
            curr_m = batch_df["chosen_variant"].astype(str) == m
            denom = int(prev_m.sum())
            retention = float((prev_m & curr_m).sum() / denom) if denom else float("nan")
            per_model_stability.append(
                {
                    "variant": m,
                    "batches_as_champion": n_pick,
                    "consecutive_same_model_pairs": cont,
                    "retention_after_self_rate": retention,
                },
            )

    summary_acc = (
        per_step_df.groupby("model", dropna=False)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "seq_oos_accuracy", "count": "n_predictions"})
        .reset_index()
        if not per_step_df.empty
        else pd.DataFrame()
    )

    return {
        "per_step_predictions": per_step_df,
        "batch_summary": batch_df,
        "summary_by_model": summary_acc,
        "stability": {
            "global_batch_stability_rate": stability_global,
            "n_batches": int(len(batch_df)),
            "definition": (
                "Fraction of batch boundaries (except first) where chosen_variant equals "
                "previous batch's chosen_variant."
            ),
        },
        "per_model_stability": pd.DataFrame(per_model_stability),
        "oos_parts": oos_parts,
        "diagnostics": diag,
        "params": {
            "fit_window_bars": tw,
            "inner_train_n": itr,
            "inner_te_n": ite,
            "batch_oos_size": B,
            "expanding_pool": expanding_pool,
            "history_cap_bars": cap if expanding_pool else None,
            "history_bars": history_bars,
            "family_filter": family_filter,
            "sticky_batch_champion": sticky_batch_champion,
            "global_pca_components": global_pca_components,
            "n_batch_starts_attempted": len(batch_starts),
            "n_batches_completed": len(batch_df),
        },
    }
