"""
apply_classifier.py
-------------------
Apply a trained pipeline to a (large) unit_table and return predictions.

Usage
-----
    from apply_classifier import apply
    predictions = apply(unit_table, model_dir="model/")
    # predictions is the input unit_table with two extra columns:
    #   pred_label     : "OK" or "NOISE"
    #   pred_noise_prob: probability of being NOISE
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from _shared import LABEL_COLS


def apply(
    unit_table: pd.DataFrame,
    model_dir: str = "model/",
    output_csv: str | None = None,
    bc_labels_to_screen: tuple[str, ...] = ("GOOD", "MUA"),
) -> pd.DataFrame:
    """
    Parameters
    ----------
    unit_table          : table with the same feature columns used during training.
                          Only rows with bc_label in bc_labels_to_screen are scored.
    model_dir           : directory containing pipeline.joblib and meta.json.
    output_csv          : if given, save results table here.
    bc_labels_to_screen : bombcell labels to run the classifier on.

    Returns
    -------
    pd.DataFrame : subset of unit_table (bc_labels_to_screen only) with columns
                   pred_label and pred_noise_prob added.
    """
    pipe = joblib.load(os.path.join(model_dir, "pipeline.joblib"))
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)

    feat_cols = meta["feature_cols"]
    classes   = np.array(meta["classes"])          # e.g. ["NOISE", "OK"] or ["OK", "NOISE"]
    noise_idx = list(classes).index("NOISE")

    # ── filter to screenable units ────────────────────────────────────────────
    screen = unit_table[unit_table["bc_label"].isin(bc_labels_to_screen)].copy()
    print(f"  Screening {len(screen):,} units  "
          f"({', '.join(bc_labels_to_screen)}) ...")

    # ── impute any columns missing from this table ────────────────────────────
    missing = [c for c in feat_cols if c not in screen.columns]
    if missing:
        print(f"  Warning: {len(missing)} feature columns missing → filled with NaN: {missing}")
        for c in missing:
            screen[c] = np.nan

    X = screen[feat_cols].values
    screen["pred_label"]      = classes[pipe.predict(X)]
    screen["pred_noise_prob"] = pipe.predict_proba(X)[:, noise_idx]

    n_noise = (screen["pred_label"] == "NOISE").sum()
    print(f"  Predicted NOISE: {n_noise:,} / {len(screen):,} "
          f"({100 * n_noise / len(screen):.1f} %)")

    if output_csv:
        cols_out = LABEL_COLS + ["bc_label", "pred_label", "pred_noise_prob"]
        screen[cols_out].to_csv(output_csv, index=False)
        print(f"  Saved predictions → {output_csv}")

    return screen
