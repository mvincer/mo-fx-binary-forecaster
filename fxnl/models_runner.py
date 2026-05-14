"""Fit many classifiers: in-sample vs OOS (delegates to :mod:`fxnl.model_fit_core`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from fxnl.model_fit_core import fit_models_core, preprocess_direction_xy


def run_direction_models(
    dset: dict[str, Any],
    *,
    random_state: int = 42,
    global_pca_components: int | None = None,
    skip_lstm: bool = False,
    models_include: list[str] | None = None,
) -> pd.DataFrame:
    """Expect keys from :func:`fxnl.data_panel.build_direction_dataset`."""
    df, _ = run_direction_models_with_preds(
        dset,
        random_state=random_state,
        global_pca_components=global_pca_components,
        skip_lstm=skip_lstm,
        models_include=models_include,
    )
    return df


def run_direction_models_with_preds(
    dset: dict[str, Any],
    *,
    random_state: int = 42,
    global_pca_components: int | None = None,
    skip_lstm: bool = False,
    models_include: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Return metrics table and test-set probability dicts (``model`` → ``proba_te``, ``pred_te``)."""
    if dset.get("error"):
        return pd.DataFrame([{"error": dset["error"]}]), {}

    X_train = dset["X_train"]
    X_test = dset["X_test"]
    y_train = dset["y_train"].to_numpy()
    y_test = dset["y_test"].to_numpy()

    pre = preprocess_direction_xy(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=random_state,
        global_pca_components=global_pca_components,
    )
    df, preds, _ = fit_models_core(
        pre,
        pair=str(dset.get("primary_id", "")),
        lead=int(dset["lead"]),
        target_mode=str(dset["target_mode"]),
        n_train=len(X_train),
        n_test=len(X_test),
        random_state=random_state,
        skip_lstm=skip_lstm,
        models_include=models_include,
    )
    return df, preds
