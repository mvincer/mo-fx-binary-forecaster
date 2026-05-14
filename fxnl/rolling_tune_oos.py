"""
Rolling walk-forward with **per-step hyperparameter search** and **1-bar-ahead** predictions.

For each forecast origin ``t``:
- ``S_all = X.iloc[t - history_bars : t]`` — training pool (default 1000 rows before ``t``).
- ``S_tune = X.iloc[t - fit_window : t]`` — inner window (e.g. 250 rows) for train/val split and grid search.
- Pick best params by **validation accuracy** (hold-out inside ``S_tune``), then refit on **all of** ``S_all``
  with those params, and predict one row: ``X.iloc[[t]]`` vs ``y.iloc[t]``.

This is designed for **speed**: small Grids, sklearn-native models, optional XGB/LGBM with tiny grids.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed

from fxnl.backtest import oos_rolling_block_stats, summarize_rolling_blocks

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def _impute_tune_split(
    X_tr: np.ndarray, X_va: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Imputer + scaler fit **only** on inner train rows (for tuning scores)."""
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_tr)
    Xva = imp.transform(X_va)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)
    return Xtr, Xva


def _impute_deploy(X_full: np.ndarray, X_one: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Imputer + scaler fit on full history pool ``S_all`` before predicting ``X_one``."""
    imp = SimpleImputer(strategy="median")
    Xf = imp.fit_transform(X_full)
    Xo = imp.transform(X_one)
    sc = StandardScaler()
    Xf = sc.fit_transform(Xf)
    Xo = sc.transform(Xo)
    return Xf, Xo


def _impute_train_test(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Imputer + scaler fit on train only; transform test (held-out)."""
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_tr)
    Xte = imp.transform(X_te)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    return Xtr, Xte


def _prep_tune_va(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    global_pca_components: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match :func:`fxnl.model_fit_core.preprocess_direction_xy` inner-train path (optional PCA + scale)."""
    if global_pca_components is None or int(global_pca_components) <= 0:
        return _impute_tune_split(X_tr, X_va)
    k = int(global_pca_components)
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_tr)
    Xva = imp.transform(X_va)
    sc0 = StandardScaler()
    Ztr = sc0.fit_transform(Xtr)
    Zva = sc0.transform(Xva)
    max_c = min(k, Ztr.shape[1], max(1, Ztr.shape[0] - 2))
    pca = PCA(n_components=max_c, svd_solver="full", random_state=int(random_state))
    Ptr = pca.fit_transform(Ztr)
    Pva = pca.transform(Zva)
    sc1 = StandardScaler()
    return sc1.fit_transform(Ptr), sc1.transform(Pva)


def _prep_train_te(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    global_pca_components: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if global_pca_components is None or int(global_pca_components) <= 0:
        return _impute_train_test(X_tr, X_te)
    k = int(global_pca_components)
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_tr)
    Xte = imp.transform(X_te)
    sc0 = StandardScaler()
    Ztr = sc0.fit_transform(Xtr)
    Zte = sc0.transform(Xte)
    max_c = min(k, Ztr.shape[1], max(1, Ztr.shape[0] - 2))
    pca = PCA(n_components=max_c, svd_solver="full", random_state=int(random_state))
    Ptr = pca.fit_transform(Ztr)
    Pte = pca.transform(Zte)
    sc1 = StandardScaler()
    return sc1.fit_transform(Ptr), sc1.transform(Pte)


def _prep_deploy_full_one(
    X_full: np.ndarray,
    X_one: np.ndarray,
    global_pca_components: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit chain on history pool ``S_all`` (PCA fit on full pool when enabled)."""
    if global_pca_components is None or int(global_pca_components) <= 0:
        return _impute_deploy(X_full, X_one)
    k = int(global_pca_components)
    imp = SimpleImputer(strategy="median")
    Xf = imp.fit_transform(X_full)
    Xo = imp.transform(X_one)
    sc0 = StandardScaler()
    Zf = sc0.fit_transform(Xf)
    Zo = sc0.transform(Xo)
    max_c = min(k, Zf.shape[1], max(1, Zf.shape[0] - 2))
    pca = PCA(n_components=max_c, svd_solver="full", random_state=int(random_state))
    Pf = pca.fit_transform(Zf)
    Po = pca.transform(Zo)
    sc1 = StandardScaler()
    return sc1.fit_transform(Pf), sc1.transform(Po)


def _fit_predict_lr(
    C: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, LogisticRegression]]:
    def _fit(
        X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
        lr = LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        lr.fit(X_train, y_train)
        return lr.predict_proba(X_pred), lr.predict(X_pred), lr

    return _fit


def _fit_predict_hgb(
    max_depth: int,
    max_iter: int,
    learning_rate: float,
    *,
    n_class: int,
    balanced: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, Any]]:
    def _fit(
        X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        kw: dict[str, Any] = dict(
            max_depth=max_depth,
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=42,
        )
        if balanced:
            try:
                kw["class_weight"] = "balanced"
            except Exception:
                pass
        clf = HistGradientBoostingClassifier(**kw)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_pred), clf.predict(X_pred), clf

    return _fit


def _fit_predict_rf(
    max_depth: int,
    n_estimators: int,
    *,
    n_class: int,
    n_jobs: int = -1,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, Any]]:
    def _fit(
        X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_pred), clf.predict(X_pred), clf

    return _fit


def _fit_predict_xgb(
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    *,
    n_class: int,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, Any]] | None:
    if xgb is None:
        return None

    def _fit(
        X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob" if n_class > 2 else "binary:logistic",
            num_class=int(n_class) if n_class > 2 else None,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss" if n_class > 2 else "logloss",
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_pred), clf.predict(X_pred), clf

    return _fit


def _fit_predict_lgb(
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    *,
    n_class: int,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, Any]] | None:
    if lgb is None:
        return None

    def _fit(
        X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        kw: dict[str, Any] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
            class_weight="balanced",
        )
        if n_class > 2:
            kw["objective"] = "multiclass"
            kw["num_class"] = int(n_class)
        else:
            kw["objective"] = "binary"
        clf = lgb.LGBMClassifier(**kw)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_pred), clf.predict(X_pred), clf

    return _fit


def _param_grids(
    n_class: int,
    *,
    rf_n_jobs: int = -1,
) -> list[tuple[str, list[tuple[str, float | int | str]], Any]]:
    """(display_name, grid of param tuples for key, factory)."""
    out: list[tuple[str, list[tuple[str, float | int | str]], Any]] = []

    lr_cs = [(f"C={c}", c) for c in (0.25, 1.0, 4.0)]
    for key, c in lr_cs:
        out.append((f"LogReg {key}", [(key, c)], _fit_predict_lr(float(c))))

    hgb_grid = []
    for md in (4, 7):
        for mi in (100, 160):
            for lr in (0.06, 0.1):
                key = f"HGB md={md} mi={mi} lr={lr}"
                hgb_grid.append((key, (md, mi, lr)))
    for key, (md, mi, lr) in hgb_grid:
        out.append(
            (
                key,
                [(key, f"{md},{mi},{lr}")],
                _fit_predict_hgb(md, mi, lr, n_class=n_class, balanced=True),
            ),
        )

    for md in (8, 12):
        for ne in (120, 200):
            key = f"RF md={md} n={ne}"
            out.append(
                (
                    key,
                    [(key, f"{md},{ne}")],
                    _fit_predict_rf(md, ne, n_class=n_class, n_jobs=rf_n_jobs),
                ),
            )

    xgb_opts = [(4, 120, 0.08), (6, 160, 0.06)]
    for md, ne, lr in xgb_opts:
        fn = _fit_predict_xgb(md, ne, lr, n_class=n_class)
        if fn is not None:
            key = f"XGB md={md} n={ne} lr={lr}"
            out.append((key, [(key, f"{md},{ne},{lr}")], fn))

    lgb_opts = [(6, 160, 0.06), (8, 200, 0.05)]
    for md, ne, lr in lgb_opts:
        fn = _fit_predict_lgb(md, ne, lr, n_class=n_class)
        if fn is not None:
            key = f"LGBM md={md} n={ne} lr={lr}"
            out.append((key, [(key, f"{md},{ne},{lr}")], fn))

    return out


def _param_grids_faster(n_class: int, *, rf_n_jobs: int = -1) -> list[tuple[str, list[tuple[str, Any]], Any]]:
    """Tiny grid for interactive runs: ~5 fits per origin instead of ~15+."""
    out: list[tuple[str, list[tuple[str, Any]], Any]] = []
    out.append(("LogReg C=1.0", [("C=1.0", 1.0)], _fit_predict_lr(1.0)))
    out.append(
        (
            "HGB md=4 mi=80 lr=0.1",
            [("HGB md=4 mi=80 lr=0.1", "4,80,0.1")],
            _fit_predict_hgb(4, 80, 0.1, n_class=n_class, balanced=True),
        ),
    )
    out.append(
        (
            "RF md=8 n=80",
            [("RF md=8 n=80", "8,80")],
            _fit_predict_rf(8, 80, n_class=n_class, n_jobs=rf_n_jobs),
        ),
    )
    return out


def _eval_tune_variant_row(
    Xtr: np.ndarray,
    y_tr: np.ndarray,
    Xva: np.ndarray,
    y_va: np.ndarray,
    display_name: str,
    fit_fn: Any,
) -> tuple[str, float, str, Any] | None:
    """One grid point; module-level for joblib pickling."""
    prefix = display_name.split()[0]
    try:
        proba_va, pred_va, _ = fit_fn(Xtr, y_tr, Xva)
        val_acc = float(accuracy_score(y_va, pred_va))
        return (prefix, val_acc, display_name, fit_fn)
    except Exception as e:
        logger.debug("tune variant %s failed: %s", display_name, e)
        return None


def _grid_variants_for_family(
    family: str,
    n_class: int,
    *,
    rf_n_jobs: int = -1,
) -> list[tuple[str, list[tuple[str, Any]], Any]]:
    """Restrict grids when user selects a single estimator family."""
    fam = family.strip().lower()
    if fam == "faster":
        return _param_grids_faster(n_class, rf_n_jobs=rf_n_jobs)
    all_grids = _param_grids(n_class, rf_n_jobs=rf_n_jobs)
    if fam in ("all", "", "none"):
        return [(a[0], a[1], a[2]) for a in all_grids]
    if fam == "fast":
        out = []
        for name, keys, fit_fn in all_grids:
            nl = name.lower()
            if nl.startswith(("logreg", "hgb", "rf")):
                out.append((name, keys, fit_fn))
        return out if out else [(a[0], a[1], a[2]) for a in all_grids]
    out = []
    for name, keys, fit_fn in all_grids:
        nl = name.lower()
        ok = (
            (fam == "logreg" and nl.startswith("logreg"))
            or (fam == "hgb" and nl.startswith("hgb"))
            or (fam == "rf" and nl.startswith("rf"))
            or (fam == "xgb" and nl.startswith("xgb"))
            or (fam in ("lgb", "lgbm") and nl.startswith("lgbm"))
        )
        if ok:
            out.append((name, keys, fit_fn))
    return out if out else [(a[0], a[1], a[2]) for a in all_grids]


def rolling_tune_sequential_oos(
    X: pd.DataFrame,
    y: pd.Series,
    forward_r: pd.Series,
    *,
    history_bars: int,
    fit_window_bars: int,
    train_frac: float,
    step: int = 1,
    random_state: int = 42,
    max_steps: int | None = None,
    family_filter: str = "all",
    expanding_pool: bool = False,
    history_cap_bars: int | None = None,
    inner_test_frac: float = 0.0,
    sticky_champion: bool = False,
    rolling_metrics_chunk: int = 100,
    tune_n_jobs: int = -1,
    global_pca_components: int | None = None,
) -> dict[str, Any]:
    """
    Sequential one-step-ahead predictions with inner tuning on ``fit_window_bars``.

    Parameters
    ----------
    history_bars
        **Fixed pool mode:** rows ``[t - history_bars, t)`` for refit. Ignored as minimum length when
        ``expanding_pool=True`` (use ``history_cap_bars`` / ``history_bars`` as cap instead).
    fit_window_bars
        Last ``W`` rows before ``t`` for inner train/val/(test) — i.e. ``[t-W, t)``.
    train_frac
        Fraction of the **non-test** part of the tune window used for training (validation gets the rest).
    expanding_pool
        If True, refit pool is **``X[max(0, t-cap):t]``** with ``cap = history_cap_bars or history_bars``,
        so the first forecast index is **``t0 = fit_window_bars``** (not ``history_bars``). Use this for
        "1000 bars total, 250 tune window, 750 steps" without needing more than 1000 rows.
    inner_test_frac
        If > 0, hold out this fraction of the **tune window** as a final **test** set; among family winners,
        the **single** model with highest **test** accuracy is deployed (instead of one model per family).
    sticky_champion
        If True, keep the previous step's winning **variant** if its test accuracy **≥** the best new variant's.
    rolling_metrics_chunk
        Block size for mean/std/min/max of Sharpe (etc.) over sequential OOS returns (0 = skip).
    tune_n_jobs
        If ``> 1``, evaluate grid variants in parallel (joblib). Use ``1`` to force sequential tuning.
        ``-1`` uses ``min(16, max(1, cpu_count()))`` workers. RandomForest uses ``n_jobs=1`` when parallel
        tuning is on to avoid oversubscribing CPU.
    global_pca_components
        If set (e.g. ``5``), apply the same impute → scale → PCA → scale pipeline as walk-forward
        (PCA fit on inner train for tuning / inner test scoring; fit on full ``S_all`` at deploy).
    """
    if train_frac <= 0 or train_frac >= 1:
        return {"error": "train_frac must be in (0, 1)"}
    if fit_window_bars < 40:
        return {"error": "fit_window_bars should be at least 40"}
    step = max(1, int(step))

    cap = int(history_cap_bars) if history_cap_bars is not None else int(history_bars)
    if expanding_pool:
        if cap < 80:
            return {"error": "expanding_pool: cap (history_cap_bars or history_bars) should be at least 80"}
        if fit_window_bars > cap:
            return {"error": "fit_window_bars cannot exceed history cap when using expanding_pool"}
    else:
        if history_bars < 80:
            return {"error": "history_bars should be at least 80"}
        if fit_window_bars > history_bars:
            return {"error": "fit_window_bars cannot exceed history_bars"}

    itf = float(inner_test_frac)
    if itf < 0 or itf >= 0.5:
        return {"error": "inner_test_frac must be in [0, 0.5)"}

    # Inner segment sizes (within tune window, oldest -> newest)
    tw = int(fit_window_bars)
    if itf > 1e-8:
        te_n = max(5, int(round(tw * itf)))
        te_n = min(te_n, tw - 20)
        rest = tw - te_n
        inner_train_n = max(10, int(round(rest * train_frac)))
        inner_val_n = rest - inner_train_n
        if inner_val_n < 5:
            inner_val_n = 5
            inner_train_n = max(10, rest - 5)
        if inner_train_n < 10 or inner_val_n < 5 or te_n < 5:
            return {
                "error": f"bad inner 3-way split: train={inner_train_n} val={inner_val_n} test={te_n} for tw={tw}",
            }
    else:
        inner_train_n = int(round(tw * train_frac))
        inner_val_n = tw - inner_train_n
        te_n = 0
        if inner_train_n < 15 or inner_val_n < 5:
            return {
                "error": f"bad inner split: train={inner_train_n} val={inner_val_n} for fit_window={tw}",
            }

    X = X.copy().sort_index()
    y = y.reindex(X.index)
    forward_r = forward_r.reindex(X.index)
    row_dates = pd.DatetimeIndex(pd.to_datetime(X.index))

    n = len(X)
    X_np = np.asarray(X, dtype=float)
    y_np_full = np.asarray(y, dtype=float)

    if expanding_pool:
        t0 = tw
    else:
        t0 = int(history_bars)

    origins = list(range(t0, n, step))
    if max_steps is not None and int(max_steps) > 0:
        origins = origins[: int(max_steps)]

    if not origins:
        return {
            "error": "rolling_tune_no_forecast_origins",
            "per_step_predictions": pd.DataFrame(),
            "oos_parts": [],
            "n_rows": n,
            "history_bars": history_bars,
            "fit_window_bars": fit_window_bars,
            "hint": (
                "Need strictly more rows than ``history_bars`` (first forecast starts at index ``history_bars``). "
                "Reduce history pool / fit window in the UI, or use a longer sample (`period` / `study_bars`)."
            ),
        }

    per_step_rows: list[dict[str, Any]] = []
    oos_parts: list[dict[str, Any]] = []
    diag = {
        "n_origins_attempted": len(origins),
        "skipped_tune_window_single_class": 0,
        "skipped_pool_single_class": 0,
        "skipped_tune_all_models_failed": 0,
        "skipped_deploy_all_failed": 0,
    }

    _ = random_state  # reserved for estimators that accept random_state
    champion_display_name: str | None = None

    tnj = int(tune_n_jobs)
    if tnj == -1:
        eff_tune_jobs = max(1, min(16, os.cpu_count() or 4))
    elif tnj <= 0:
        eff_tune_jobs = 1
    else:
        eff_tune_jobs = max(1, min(32, tnj))
    parallel_tune = eff_tune_jobs > 1
    rf_n_jobs = 1 if parallel_tune else -1

    for _ti, t in enumerate(origins):
        prev_champion = champion_display_name

        if expanding_pool:
            start_all = max(0, t - cap)
            S_all_X = X_np[start_all:t]
            S_all_y_raw = y_np_full[start_all:t]
        else:
            hb = int(history_bars)
            S_all_X = X_np[t - hb : t]
            S_all_y_raw = y_np_full[t - hb : t]

        S_tune_X = X_np[t - tw : t]
        S_tune_y_raw = y_np_full[t - tw : t]

        if np.unique(S_tune_y_raw.astype(int)).size < 2:
            diag["skipped_tune_window_single_class"] += 1
            continue

        X_one = X_np[t : t + 1]
        y_t_raw = float(y_np_full[t])
        r_t = float(forward_r.iloc[t])

        u_labels = np.sort(
            np.unique(np.concatenate([S_all_y_raw.ravel(), np.array([y_t_raw])]).astype(int)),
        )
        to_idx = {int(v): i for i, v in enumerate(u_labels)}
        n_cls = int(len(u_labels))
        S_all_y_map = np.array([to_idx[int(v)] for v in S_all_y_raw], dtype=int)
        S_tune_y_map = np.array([to_idx[int(v)] for v in S_tune_y_raw], dtype=int)
        y_t_m = int(to_idx[int(y_t_raw)])

        if n_cls < 2:
            diag["skipped_pool_single_class"] += 1
            continue

        if te_n > 0:
            i_va_e = inner_train_n + inner_val_n
            X_tr_in = S_tune_X[:inner_train_n]
            X_va_in = S_tune_X[inner_train_n:i_va_e]
            X_te_in = S_tune_X[i_va_e:]
            y_tr_in = S_tune_y_map[:inner_train_n]
            y_va_in = S_tune_y_map[inner_train_n:i_va_e]
            y_te_in = S_tune_y_map[i_va_e:]
        else:
            X_tr_in = S_tune_X[:inner_train_n]
            X_va_in = S_tune_X[inner_train_n:]
            y_tr_in = S_tune_y_map[:inner_train_n]
            y_va_in = S_tune_y_map[inner_train_n:]

        Xtr_tune, Xva_tune = _prep_tune_va(
            X_tr_in, X_va_in, global_pca_components, random_state
        )
        Xtr_inner_te: np.ndarray | None = None
        Xte_inner: np.ndarray | None = None
        if te_n > 0:
            Xtr_inner_te, Xte_inner = _prep_train_te(
                X_tr_in, X_te_in, global_pca_components, random_state
            )

        grids = _grid_variants_for_family(family_filter, n_cls, rf_n_jobs=rf_n_jobs)
        best_by_display: dict[str, tuple[float, str, Any]] = {}

        if parallel_tune:
            rows = Parallel(n_jobs=eff_tune_jobs, batch_size=1)(
                delayed(_eval_tune_variant_row)(
                    Xtr_tune,
                    y_tr_in,
                    Xva_tune,
                    y_va_in,
                    display_name,
                    fit_fn,
                )
                for display_name, _keys, fit_fn in grids
            )
            for row in rows:
                if row is None:
                    continue
                prefix, val_acc, display_name, fit_fn = row
                prev = best_by_display.get(prefix)
                if prev is None or val_acc > prev[0]:
                    best_by_display[prefix] = (val_acc, display_name, fit_fn)
        else:
            for display_name, _keys, fit_fn in grids:
                prefix = display_name.split()[0]
                try:
                    proba_va, pred_va, _ = fit_fn(Xtr_tune, y_tr_in, Xva_tune)
                    val_acc = float(accuracy_score(y_va_in, pred_va))

                    prev = best_by_display.get(prefix)
                    if prev is None or val_acc > prev[0]:
                        best_by_display[prefix] = (val_acc, display_name, fit_fn)
                except Exception as e:
                    logger.debug("t=%s model_line=%s failed: %s", t, display_name, e)
                    continue

        if not best_by_display:
            diag["skipped_tune_all_models_failed"] += 1
            continue

        def _acc_on_test(fit_fn_inner: Any) -> float:
            if te_n <= 0 or Xtr_inner_te is None or Xte_inner is None:
                return float("nan")
            _pr, pred_te, _ = fit_fn_inner(Xtr_inner_te, y_tr_in, Xte_inner)
            return float(accuracy_score(y_te_in, pred_te))

        def _val_acc_for(fit_fn_inner: Any) -> float:
            _pr, pred_va, _ = fit_fn_inner(Xtr_tune, y_tr_in, Xva_tune)
            return float(accuracy_score(y_va_in, pred_va))

        deploy_items: list[tuple[str, str, str, Any, float, float]] = []
        # (label_name, prefix, display_name, fit_fn, val_acc, test_acc or nan)
        sticky_used_this_step = False
        use_display_final: str | None = None

        if te_n > 0:
            scored: list[tuple[float, float, str, str, Any]] = []
            for prefix, (val_acc, display_name, fit_fn) in best_by_display.items():
                try:
                    ta = _acc_on_test(fit_fn)
                    scored.append((ta, val_acc, prefix, display_name, fit_fn))
                except Exception as e:
                    logger.debug("test score t=%s %s: %s", t, display_name, e)
            if not scored:
                diag["skipped_tune_all_models_failed"] += 1
                continue
            scored.sort(key=lambda x: x[0], reverse=True)
            best_new_test, _val_acc_b, prefix_b, dn_b, fn_b = scored[0]

            use_display = dn_b
            use_fn = fn_b
            use_prefix = prefix_b
            use_test = best_new_test

            if sticky_champion and prev_champion:
                ch_fn = None
                for dn_s, _ks, fn_s in grids:
                    if dn_s == prev_champion:
                        ch_fn = fn_s
                        break
                if ch_fn is not None:
                    try:
                        ch_test = _acc_on_test(ch_fn)
                        if ch_test >= best_new_test - 1e-15:
                            use_display = prev_champion
                            use_fn = ch_fn
                            use_prefix = use_display.split()[0]
                            use_test = ch_test
                    except Exception as e:
                        logger.debug("sticky champion t=%s: %s", t, e)

            use_val = _val_acc_for(use_fn)
            sticky_used_this_step = bool(
                sticky_champion
                and prev_champion is not None
                and use_display == prev_champion
                and dn_b != prev_champion
            )
            use_display_final = use_display
            label_name = f"{use_prefix} (tuned)"
            deploy_items.append((label_name, use_prefix, use_display, use_fn, use_val, use_test))
        else:
            for prefix, (val_acc, display_name, fit_fn) in best_by_display.items():
                deploy_items.append(
                    (f"{prefix} (tuned)", prefix, display_name, fit_fn, val_acc, float("nan")),
                )

        step_preds: dict[str, dict[str, np.ndarray]] = {}
        for label_name, prefix, display_name, fit_fn, val_acc, test_acc in deploy_items:
            try:
                Xfull, Xpred = _prep_deploy_full_one(
                    S_all_X, X_one, global_pca_components, random_state
                )
                proba_o, pred_o, _ = fit_fn(Xfull, S_all_y_map, Xpred)
                proba_o = np.asarray(proba_o, dtype=float)
                if proba_o.ndim == 1:
                    proba_o = proba_o.reshape(1, -1)

                pred_mapped = int(np.argmax(proba_o, axis=1)[0])
                correct = int(pred_mapped == y_t_m)

                row_rec: dict[str, Any] = {
                    "forecast_idx": t,
                    "date": row_dates[t],
                    "model": label_name,
                    "y_true": y_t_m,
                    "y_true_raw": int(y_t_raw),
                    "pred_class": pred_mapped,
                    "correct": correct,
                    "inner_val_accuracy": val_acc,
                    "chosen_variant": display_name,
                    "forward_r": r_t,
                }
                if te_n > 0:
                    row_rec["inner_test_accuracy"] = test_acc
                    row_rec["sticky_champion"] = sticky_used_this_step
                per_step_rows.append(row_rec)

                step_preds[label_name] = {
                    "proba_te": proba_o,
                    "pred_te": np.array([pred_mapped], dtype=int),
                }
            except Exception as e:
                logger.debug("refit t=%s %s: %s", t, prefix, e)
                continue

        if not step_preds:
            diag["skipped_deploy_all_failed"] += 1
            continue

        if te_n > 0 and use_display_final is not None:
            champion_display_name = use_display_final

        oos_parts.append(
            {
                "start": t,
                "end": t + 1,
                "dates_te": pd.DatetimeIndex([row_dates[t]]),
                "y_te": np.array([y_t_m], dtype=int),
                "forward_r_te": np.array([r_t], dtype=float),
                "preds": step_preds,
            },
        )

    if not oos_parts:
        hint_parts = []
        if diag["skipped_tune_window_single_class"]:
            hint_parts.append(
                "inner tune window had only one class label (try smaller fit window or different lead/target)."
            )
        if diag["skipped_pool_single_class"]:
            hint_parts.append(
                "history pool plus current y still had only one class (need class variation in labels)."
            )
        if diag["skipped_tune_all_models_failed"]:
            hint_parts.append(
                "every tuner variant raised on inner split (often NaN/constant features — reduce columns or check data)."
            )
        if diag["skipped_deploy_all_failed"]:
            hint_parts.append(
                "tuning succeeded but refit/predict failed on history pool (see logs at DEBUG)."
            )
        if not hint_parts:
            hint_parts.append("no forecast origins produced a valid step (unexpected — check row count vs history_bars).")
        return {
            "error": "no_successful_steps",
            "per_step_predictions": pd.DataFrame(),
            "oos_parts": [],
            "diagnostics": diag,
            "hint": " ".join(hint_parts),
        }

    per_step_df = pd.DataFrame(per_step_rows)

    summary_acc = (
        per_step_df.groupby("model", dropna=False)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "seq_oos_accuracy", "count": "n_predictions"})
        .reset_index()
        .sort_values("seq_oos_accuracy", ascending=False)
    )

    rolling_blocks_by_model: dict[str, list[dict[str, float]]] = {}
    rolling_block_meta_by_model: dict[str, dict[str, Any]] = {}
    rolling_summary_by_model: dict[str, dict[str, dict[str, float]]] = {}
    chunk_sz = int(rolling_metrics_chunk)
    if chunk_sz > 0 and not per_step_df.empty and "forward_r" in per_step_df.columns:
        for model_name in per_step_df["model"].unique():
            sub = per_step_df.loc[per_step_df["model"] == model_name].sort_values("forecast_idx")
            strat = np.where(
                sub["correct"].astype(bool),
                pd.to_numeric(sub["forward_r"], errors="coerce"),
                -pd.to_numeric(sub["forward_r"], errors="coerce"),
            )
            strat_s = pd.Series(strat, index=pd.DatetimeIndex(pd.to_datetime(sub["date"])))
            blocks, rmeta = oos_rolling_block_stats(strat_s, None, chunk_size=chunk_sz)
            rolling_blocks_by_model[str(model_name)] = blocks
            rolling_block_meta_by_model[str(model_name)] = rmeta
            rolling_summary_by_model[str(model_name)] = summarize_rolling_blocks(blocks)

    return {
        "per_step_predictions": per_step_df,
        "summary_by_model": summary_acc,
        "oos_parts": oos_parts,
        "rolling_block_summary_by_model": rolling_summary_by_model,
        "rolling_blocks_by_model": rolling_blocks_by_model,
        "rolling_block_meta_by_model": rolling_block_meta_by_model,
        "params": {
            "history_bars": history_bars,
            "fit_window_bars": fit_window_bars,
            "train_frac": train_frac,
            "step": step,
            "inner_train_n": inner_train_n,
            "inner_val_n": inner_val_n,
            "inner_test_n": te_n,
            "inner_test_frac": itf,
            "expanding_pool": expanding_pool,
            "history_cap_bars": cap if expanding_pool else None,
            "n_forecast_origins": len(oos_parts),
            "family_filter": family_filter,
            "sticky_champion": sticky_champion,
            "rolling_metrics_chunk": chunk_sz,
            "tune_n_jobs_requested": tnj,
            "tune_n_jobs_effective": eff_tune_jobs,
            "parallel_tune": parallel_tune,
            "global_pca_components": int(global_pca_components)
            if global_pca_components is not None and int(global_pca_components) > 0
            else None,
        },
    }
