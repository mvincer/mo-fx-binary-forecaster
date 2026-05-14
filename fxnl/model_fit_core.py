"""Preprocess + fit all direction classifiers; collect test probabilities for ensembles & walk-forward."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings

from fxnl.lstm_torch import fit_predict_lstm
from fxnl.metrics_eval import eval_split, flatten_result_row

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None  # type: ignore

# Row ``model`` names used in :func:`fit_models_core` (for UI / ``models_include`` filter).
DIRECTION_MODEL_OPTIONS: tuple[str, ...] = (
    "Logistic (sklearn, on global PCA)",
    "PCA + logistic (sklearn)",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "LSTM (PyTorch)",
    "Logit (statsmodels, PCA)",
    "MNLogit (statsmodels, PCA)",
    "Probit (statsmodels, PCA)",
    "Random forest",
    "Gradient boosting (sklearn)",
    "Hist gradient boosting",
    "SVM (RBF)",
    "MLP neural net",
    "Ensemble (mean proba)",
    "Ensemble (weighted train acc)",
)


def normalize_models_include(models_include: Iterable[str] | None) -> frozenset[str] | None:
    """``None`` = all models; else only listed ``model`` names are fitted (ensembles need member models)."""
    if models_include is None:
        return None
    s = frozenset(str(x).strip() for x in models_include if str(x).strip())
    return None if len(s) == 0 else s


def _imputed(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X_train), imp.transform(X_test)


def _scaled(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


def preprocess_direction_xy(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int,
    global_pca_components: int | None,
) -> dict[str, Any]:
    labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    n_classes = len(labels)
    label_to_idx = {int(v): i for i, v in enumerate(labels)}
    if set(labels) != set(range(n_classes)):
        y_train = np.array([label_to_idx[int(v)] for v in y_train], dtype=int)
        y_test = np.array([label_to_idx[int(v)] for v in y_test], dtype=int)

    Xi_tr, Xi_te = _imputed(X_train, X_test)
    use_global_pca = global_pca_components is not None and int(global_pca_components) > 0
    global_pca_var: float | None = None
    if use_global_pca:
        sc0 = StandardScaler()
        Z_tr = sc0.fit_transform(Xi_tr)
        Z_te = sc0.transform(Xi_te)
        max_c = min(int(global_pca_components), Z_tr.shape[1], max(1, Z_tr.shape[0] - 2))
        pca_g = PCA(n_components=max_c, svd_solver="full", random_state=random_state)
        Xi_tr = pca_g.fit_transform(Z_tr)
        Xi_te = pca_g.transform(Z_te)
        global_pca_var = float(np.sum(pca_g.explained_variance_ratio_))

    n_feat_report = int(Xi_tr.shape[1])
    Xs_tr, Xs_te = _scaled(Xi_tr, Xi_te)

    return {
        "Xi_tr": Xi_tr,
        "Xi_te": Xi_te,
        "Xs_tr": Xs_tr,
        "Xs_te": Xs_te,
        "y_train": y_train,
        "y_test": y_test,
        "n_classes": n_classes,
        "n_feat_report": n_feat_report,
        "use_global_pca": use_global_pca,
        "global_pca_var": global_pca_var,
        "n_feat_raw": int(X_train.shape[1]),
    }


def fit_models_core(
    pre: dict[str, Any],
    *,
    pair: str,
    lead: int,
    target_mode: str,
    n_train: int,
    n_test: int,
    random_state: int = 42,
    skip_lstm: bool = False,
    models_include: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, float]]:
    Xi_tr = pre["Xi_tr"]
    Xi_te = pre["Xi_te"]
    Xs_tr = pre["Xs_tr"]
    Xs_te = pre["Xs_te"]
    y_train = pre["y_train"]
    y_test = pre["y_test"]
    n_classes = int(pre["n_classes"])
    n_feat_report = int(pre["n_feat_report"])
    use_global_pca = bool(pre["use_global_pca"])
    global_pca_var = pre.get("global_pca_var")
    n_feat_raw = int(pre["n_feat_raw"])

    rows: list[dict[str, Any]] = []
    preds: dict[str, dict[str, np.ndarray]] = {}
    train_acc: dict[str, float] = {}
    inc = normalize_models_include(models_include)

    def want(name: str) -> bool:
        return inc is None or name in inc

    def record_pred(name: str, proba_te: np.ndarray | None, pred_te: np.ndarray) -> None:
        if proba_te is not None and np.asarray(proba_te).size > 0:
            preds[name] = {"proba_te": np.asarray(proba_te), "pred_te": np.asarray(pred_te).ravel()}

    def add_row(
        name: str,
        ytr: np.ndarray,
        yte: np.ndarray,
        pred_tr: np.ndarray,
        pred_te: np.ndarray,
        proba_tr: np.ndarray | None,
        proba_te: np.ndarray | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        tr_m = eval_split(ytr, pred_tr, proba_tr, n_classes=n_classes)
        te_m = eval_split(yte, pred_te, proba_te, n_classes=n_classes)
        train_acc[name] = float(tr_m.get("accuracy", float("nan")))
        ex = dict(extra) if extra else {}
        if use_global_pca and global_pca_var is not None:
            ex.setdefault("global_pca_k", n_feat_report)
            ex.setdefault("global_pca_var_explained", global_pca_var)
            ex.setdefault("n_features_raw", n_feat_raw)
        rows.append(
            flatten_result_row(
                name,
                pair=pair,
                lead=lead,
                target_mode=target_mode,
                n_train=len(ytr),
                n_test=len(yte),
                n_features=n_feat_report,
                train_metrics=tr_m,
                test_metrics=te_m,
                extra=ex or None,
            ),
        )
        record_pred(name, proba_te, pred_te)

    nan_metrics = {k: float("nan") for k in ("accuracy", "balanced_accuracy", "log_loss", "f1_macro", "roc_auc")}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if use_global_pca:
            if want("Logistic (sklearn, on global PCA)"):
                lr_pca = LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                )
                lr_pca.fit(Xi_tr, y_train)
                add_row(
                    "Logistic (sklearn, on global PCA)",
                    y_train,
                    y_test,
                    lr_pca.predict(Xi_tr),
                    lr_pca.predict(Xi_te),
                    lr_pca.predict_proba(Xi_tr),
                    lr_pca.predict_proba(Xi_te),
                    extra={},
                )
        else:
            if want("PCA + logistic (sklearn)"):
                pca_lr = Pipeline(
                    [
                        ("sc", StandardScaler()),
                        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=random_state)),
                        (
                            "lr",
                            LogisticRegression(
                                max_iter=2500,
                                class_weight="balanced",
                                solver="lbfgs",
                                random_state=random_state,
                            ),
                        ),
                    ],
                )
                pca_lr.fit(Xi_tr, y_train)
                pca_step = pca_lr.named_steps["pca"]
                add_row(
                    "PCA + logistic (sklearn)",
                    y_train,
                    y_test,
                    pca_lr.predict(Xi_tr),
                    pca_lr.predict(Xi_te),
                    pca_lr.predict_proba(Xi_tr),
                    pca_lr.predict_proba(Xi_te),
                    extra={
                        "pca_n_components": int(pca_step.n_components_),
                        "pca_var_explained": float(np.sum(pca_step.explained_variance_ratio_)),
                    },
                )

    if xgb is not None:
        if want("XGBoost"):
            xgb_clf = xgb.XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=int(n_classes) if n_classes > 2 else None,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
            )
            xgb_clf.fit(Xi_tr, y_train)
            add_row(
                "XGBoost",
                y_train,
                y_test,
                xgb_clf.predict(Xi_tr),
                xgb_clf.predict(Xi_te),
                xgb_clf.predict_proba(Xi_tr),
                xgb_clf.predict_proba(Xi_te),
            )
    elif inc is None or want("XGBoost"):
        rows.append(
            flatten_result_row(
                "XGBoost (skipped)",
                pair=pair,
                lead=lead,
                target_mode=target_mode,
                n_train=n_train,
                n_test=n_test,
                n_features=n_feat_report,
                train_metrics=nan_metrics.copy(),
                test_metrics=nan_metrics.copy(),
                extra={"note": "pip install xgboost"},
            ),
        )

    if lgb is not None:
        if want("LightGBM"):
            lgb_kw: dict[str, Any] = dict(
                n_estimators=300,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multiclass" if n_classes > 2 else "binary",
                random_state=random_state,
                verbose=-1,
                class_weight="balanced",
            )
            if n_classes > 2:
                lgb_kw["num_class"] = int(n_classes)
            lgb_clf = lgb.LGBMClassifier(**lgb_kw)
            lgb_clf.fit(Xi_tr, y_train)
            add_row(
                "LightGBM",
                y_train,
                y_test,
                lgb_clf.predict(Xi_tr),
                lgb_clf.predict(Xi_te),
                lgb_clf.predict_proba(Xi_tr),
                lgb_clf.predict_proba(Xi_te),
            )
    elif inc is None or want("LightGBM"):
        rows.append(
            flatten_result_row(
                "LightGBM (skipped)",
                pair=pair,
                lead=lead,
                target_mode=target_mode,
                n_train=n_train,
                n_test=n_test,
                n_features=n_feat_report,
                train_metrics=nan_metrics.copy(),
                test_metrics=nan_metrics.copy(),
                extra={"note": "pip install lightgbm"},
            ),
        )

    if CatBoostClassifier is not None:
        if want("CatBoost"):
            try:
                cbc = CatBoostClassifier(
                    iterations=400,
                    depth=6,
                    learning_rate=0.06,
                    loss_function="MultiClass" if n_classes > 2 else "Logloss",
                    random_seed=random_state,
                    verbose=False,
                    auto_class_weights="Balanced",
                )
                cbc.fit(Xi_tr, y_train)
                add_row(
                    "CatBoost",
                    y_train,
                    y_test,
                    cbc.predict(Xi_tr),
                    cbc.predict(Xi_te),
                    cbc.predict_proba(Xi_tr),
                    cbc.predict_proba(Xi_te),
                )
            except Exception as e:
                rows.append(
                    flatten_result_row(
                        "CatBoost (error)",
                        pair=pair,
                        lead=lead,
                        target_mode=target_mode,
                        n_train=n_train,
                        n_test=n_test,
                        n_features=n_feat_report,
                        train_metrics=nan_metrics.copy(),
                        test_metrics=nan_metrics.copy(),
                        extra={"note": str(e)},
                    ),
                )
    elif inc is None or want("CatBoost"):
        rows.append(
            flatten_result_row(
                "CatBoost (skipped)",
                pair=pair,
                lead=lead,
                target_mode=target_mode,
                n_train=n_train,
                n_test=n_test,
                n_features=n_feat_report,
                train_metrics=nan_metrics.copy(),
                test_metrics=nan_metrics.copy(),
                extra={"note": "pip install catboost"},
            ),
        )

    if not skip_lstm and (inc is None or want("LSTM (PyTorch)")):
        try:
            lstm_out = fit_predict_lstm(
                Xs_tr,
                y_train,
                Xs_te,
                y_test,
                n_train_rows=n_train,
                n_classes=n_classes,
                seq_len=min(24, max(8, n_train // 25)),
                epochs=40,
                random_state=random_state,
            )
            if "error" not in lstm_out:
                add_row(
                    "LSTM (PyTorch)",
                    lstm_out["y_train_true"],
                    lstm_out["y_test_true"],
                    lstm_out["y_train_pred"],
                    lstm_out["y_test_pred"],
                    lstm_out["train_proba"],
                    lstm_out["test_proba"],
                    extra={
                        "lstm_seq_len": lstm_out["seq_len"],
                        "lstm_hidden": lstm_out["hidden"],
                        "lstm_epochs": lstm_out["epochs"],
                    },
                )
            else:
                rows.append(
                    flatten_result_row(
                        "LSTM (PyTorch) (skipped)",
                        pair=pair,
                        lead=lead,
                        target_mode=target_mode,
                        n_train=n_train,
                        n_test=n_test,
                        n_features=n_feat_report,
                        train_metrics=nan_metrics.copy(),
                        test_metrics=nan_metrics.copy(),
                        extra={"note": str(lstm_out.get("error"))},
                    ),
                )
        except Exception as e:
            logger.exception("LSTM failed")
            rows.append(
                flatten_result_row(
                    "LSTM (PyTorch) (error)",
                    pair=pair,
                    lead=lead,
                    target_mode=target_mode,
                    n_train=n_train,
                    n_test=n_test,
                    n_features=n_feat_report,
                    train_metrics=nan_metrics.copy(),
                    test_metrics=nan_metrics.copy(),
                    extra={"note": str(e)},
                ),
            )

    if use_global_pca:
        sm_pca_dim = int(Xi_tr.shape[1])
        Z_tr = np.asarray(Xi_tr, dtype=float)
        Z_te = np.asarray(Xi_te, dtype=float)
    else:
        n_comp = int(min(40, Xi_tr.shape[1], max(8, Xi_tr.shape[0] // 4)))
        pca_sm = PCA(n_components=n_comp, svd_solver="full", random_state=random_state)
        Z_tr = pca_sm.fit_transform(Xs_tr)
        Z_te = pca_sm.transform(Xs_te)
        sm_pca_dim = n_comp
    Z_tr_c = sm.add_constant(Z_tr, has_constant="add")
    Z_te_c = sm.add_constant(Z_te, has_constant="add")

    if inc is None or want("Logit (statsmodels, PCA)") or want("MNLogit (statsmodels, PCA)"):
        try:
            if n_classes == 2 and (inc is None or want("Logit (statsmodels, PCA)")):
                logit_res = sm.Logit(y_train, Z_tr_c).fit(disp=False, maxiter=400)
                pr_tr = logit_res.predict(Z_tr_c)
                pr_te = logit_res.predict(Z_te_c)
                pred_tr = (pr_tr >= 0.5).astype(int)
                pred_te = (pr_te >= 0.5).astype(int)
                proba_tr = np.column_stack([1.0 - pr_tr, pr_tr])
                proba_te = np.column_stack([1.0 - pr_te, pr_te])
                add_row(
                    "Logit (statsmodels, PCA)",
                    y_train,
                    y_test,
                    pred_tr,
                    pred_te,
                    proba_tr,
                    proba_te,
                    extra={"pseudo_r2": float(logit_res.prsquared), "pca_dim": sm_pca_dim},
                )
            elif n_classes > 2 and (inc is None or want("MNLogit (statsmodels, PCA)")):
                mn_res = sm.MNLogit(y_train, Z_tr_c).fit(disp=False, maxiter=300)
                pr_tr = mn_res.predict(Z_tr_c)
                pr_te = mn_res.predict(Z_te_c)
                pred_tr = np.argmax(pr_tr.values, axis=1) if hasattr(pr_tr, "values") else np.argmax(pr_tr, axis=1)
                pred_te = np.argmax(pr_te.values, axis=1) if hasattr(pr_te, "values") else np.argmax(pr_te, axis=1)
                p_tr = pr_tr.values if hasattr(pr_tr, "values") else np.asarray(pr_tr)
                p_te = pr_te.values if hasattr(pr_te, "values") else np.asarray(pr_te)
                add_row(
                    "MNLogit (statsmodels, PCA)",
                    y_train,
                    y_test,
                    pred_tr,
                    pred_te,
                    p_tr,
                    p_te,
                    extra={"pseudo_r2": float(mn_res.prsquared), "pca_dim": sm_pca_dim},
                )
        except Exception as e:
            if inc is None or want("Logit (statsmodels, PCA)") or want("MNLogit (statsmodels, PCA)"):
                rows.append(
                    flatten_result_row(
                        "Logit / MNLogit (statsmodels) (error)",
                        pair=pair,
                        lead=lead,
                        target_mode=target_mode,
                        n_train=n_train,
                        n_test=n_test,
                        n_features=n_feat_report,
                        train_metrics=nan_metrics.copy(),
                        test_metrics=nan_metrics.copy(),
                        extra={"note": str(e)},
                    ),
                )

    if n_classes == 2:
        if inc is None or want("Probit (statsmodels, PCA)"):
            try:
                probit_res = sm.Probit(y_train, Z_tr_c).fit(disp=False, maxiter=400)
                pr_tr = probit_res.predict(Z_tr_c)
                pr_te = probit_res.predict(Z_te_c)
                pred_tr = (pr_tr >= 0.5).astype(int)
                pred_te = (pr_te >= 0.5).astype(int)
                proba_tr = np.column_stack([1.0 - pr_tr, pr_tr])
                proba_te = np.column_stack([1.0 - pr_te, pr_te])
                add_row(
                    "Probit (statsmodels, PCA)",
                    y_train,
                    y_test,
                    pred_tr,
                    pred_te,
                    proba_tr,
                    proba_te,
                    extra={"pseudo_r2": float(getattr(probit_res, "prsquared", float("nan"))), "pca_dim": sm_pca_dim},
                )
            except Exception as e:
                rows.append(
                    flatten_result_row(
                        "Probit (statsmodels) (error)",
                        pair=pair,
                        lead=lead,
                        target_mode=target_mode,
                        n_train=n_train,
                        n_test=n_test,
                        n_features=n_feat_report,
                        train_metrics=nan_metrics.copy(),
                        test_metrics=nan_metrics.copy(),
                        extra={"note": str(e)},
                    ),
                )
    elif inc is None:
        rows.append(
            flatten_result_row(
                "Probit (N/A for 3-class)",
                pair=pair,
                lead=lead,
                target_mode=target_mode,
                n_train=n_train,
                n_test=n_test,
                n_features=n_feat_report,
                train_metrics=nan_metrics.copy(),
                test_metrics=nan_metrics.copy(),
                extra={"note": "binary only"},
            ),
        )

    if want("Random forest"):
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(Xi_tr, y_train)
        add_row(
            "Random forest",
            y_train,
            y_test,
            rf.predict(Xi_tr),
            rf.predict(Xi_te),
            rf.predict_proba(Xi_tr),
            rf.predict_proba(Xi_te),
        )

    if want("Gradient boosting (sklearn)"):
        gbt = GradientBoostingClassifier(random_state=random_state, max_depth=3, n_estimators=200)
        gbt.fit(Xi_tr, y_train)
        add_row(
            "Gradient boosting (sklearn)",
            y_train,
            y_test,
            gbt.predict(Xi_tr),
            gbt.predict(Xi_te),
            gbt.predict_proba(Xi_tr),
            gbt.predict_proba(Xi_te),
        )

    if want("Hist gradient boosting"):
        try:
            hgb = HistGradientBoostingClassifier(
                max_depth=6,
                max_iter=200,
                class_weight="balanced",
                random_state=random_state,
            )
        except TypeError:
            hgb = HistGradientBoostingClassifier(
                max_depth=6,
                max_iter=200,
                random_state=random_state,
            )
        hgb.fit(Xi_tr, y_train)
        add_row(
            "Hist gradient boosting",
            y_train,
            y_test,
            hgb.predict(Xi_tr),
            hgb.predict(Xi_te),
            hgb.predict_proba(Xi_tr),
            hgb.predict_proba(Xi_te),
        )

    if want("SVM (RBF)"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state)
            svm.fit(Xs_tr, y_train)
            add_row(
                "SVM (RBF)",
                y_train,
                y_test,
                svm.predict(Xs_tr),
                svm.predict(Xs_te),
                svm.predict_proba(Xs_tr),
                svm.predict_proba(Xs_te),
            )

    if want("MLP neural net"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=400,
                random_state=random_state,
                early_stopping=True,
            )
            mlp.fit(Xs_tr, y_train)
            add_row(
                "MLP neural net",
                y_train,
                y_test,
                mlp.predict(Xs_tr),
                mlp.predict(Xs_te),
                mlp.predict_proba(Xs_tr),
                mlp.predict_proba(Xs_te),
            )

    # Ensembles from models that produced test probabilities (exclude prior ensemble keys)
    skip_ens = ("Ensemble (mean proba)", "Ensemble (weighted train acc)")
    stack_names = [
        n for n in preds if preds[n].get("proba_te") is not None and n not in skip_ens
    ]
    run_mean_ens = inc is None or want("Ensemble (mean proba)")
    run_w_ens = inc is None or want("Ensemble (weighted train acc)")
    if len(stack_names) >= 2 and (run_mean_ens or run_w_ens):
        shapes = {preds[n]["proba_te"].shape[1] for n in stack_names}
        if len(shapes) == 1:
            acc_vec = np.array([max(0.0, train_acc.get(n, 0.0)) for n in stack_names], dtype=float)
            s_acc = acc_vec.sum()
            if run_mean_ens:
                P_mean = np.mean(np.stack([preds[n]["proba_te"] for n in stack_names], axis=0), axis=0)
                pred_mean = np.argmax(P_mean, axis=1)
                te_m_mean = eval_split(y_test, pred_mean, P_mean, n_classes=n_classes)
                rows.append(
                    flatten_result_row(
                        "Ensemble (mean proba)",
                        pair=pair,
                        lead=lead,
                        target_mode=target_mode,
                        n_train=n_train,
                        n_test=n_test,
                        n_features=n_feat_report,
                        train_metrics=nan_metrics.copy(),
                        test_metrics=te_m_mean,
                        extra={"n_members": len(stack_names)},
                    ),
                )
                preds["Ensemble (mean proba)"] = {"proba_te": P_mean, "pred_te": pred_mean}
                train_acc["Ensemble (mean proba)"] = float(te_m_mean.get("accuracy", float("nan")))

            if run_w_ens and s_acc > 1e-12:
                w = acc_vec / s_acc
                P_w = np.tensordot(w, np.stack([preds[n]["proba_te"] for n in stack_names], axis=0), axes=([0], [0]))
                pred_w = np.argmax(P_w, axis=1)
                te_m_w = eval_split(y_test, pred_w, P_w, n_classes=n_classes)
                rows.append(
                    flatten_result_row(
                        "Ensemble (weighted train acc)",
                        pair=pair,
                        lead=lead,
                        target_mode=target_mode,
                        n_train=n_train,
                        n_test=n_test,
                        n_features=n_feat_report,
                        train_metrics=nan_metrics.copy(),
                        test_metrics=te_m_w,
                        extra={"n_members": len(stack_names)},
                    ),
                )
                preds["Ensemble (weighted train acc)"] = {"proba_te": P_w, "pred_te": pred_w}
                train_acc["Ensemble (weighted train acc)"] = float(te_m_w.get("accuracy", float("nan")))

    if not rows:
        rows.append(
            flatten_result_row(
                "No models (check models_include)",
                pair=pair,
                lead=lead,
                target_mode=target_mode,
                n_train=n_train,
                n_test=n_test,
                n_features=n_feat_report,
                train_metrics=nan_metrics.copy(),
                test_metrics=nan_metrics.copy(),
                extra={"models_include": sorted(inc) if inc else []},
            ),
        )

    return pd.DataFrame(rows), preds, train_acc
