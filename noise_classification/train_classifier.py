"""
train_classifier.py
-------------------
Train a Random Forest to predict NOISE vs OK on manually labeled units,
using bombcell QC metrics as features.

Usage
-----
    from train_classifier import train
    results = train(
        labels_csv   = "labels.csv",
        unit_table   = unit_table,      # full table with bc_* feature columns
        trial_table  = trial_table,     # for misclassification plots
        model_dir    = "model/",
    )
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from _shared import LABEL_COLS, get_feature_cols, plot_unit


def train(
    labels_csv: str,
    unit_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    model_dir: str = "model/",
    n_splits: int = 5,
    n_misclassified_examples: int = 6,
) -> dict:
    """
    Parameters
    ----------
    labels_csv  : CSV produced by label_gui (columns: LABEL_COLS + manual_label).
    unit_table  : full unit table including feature columns and spike_times.
    trial_table : needed only for misclassification raster plots.
    model_dir   : directory where pipeline + feature list are saved.

    Returns
    -------
    dict with cross-validated metrics and the trained pipeline.
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── load & merge ──────────────────────────────────────────────────────────
    labels = pd.read_csv(labels_csv)
    df = labels.merge(unit_table, on=LABEL_COLS, how="left")
    print(f"  Labeled units: {len(df)}  "
          f"(OK={( df['manual_label']=='OK').sum()}, "
          f"NOISE={(df['manual_label']=='NOISE').sum()})")

    feat_cols = get_feature_cols(df)
    X = df[feat_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["manual_label"])   # OK=0 or 1, NOISE=1 or 0

    # ── pipeline: impute NaN → Random Forest ──────────────────────────────────
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced", random_state=0, n_jobs=-1,
        )),
    ])

    # ── cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_results = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        return_train_score=True,
    )

    print(f"\n{'─'*55}")
    print(f"  {n_splits}-fold cross-validation")
    print(f"{'─'*55}")
    for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
        tr = cv_results[f"train_{metric}"]
        te = cv_results[f"test_{metric}"]
        print(f"  {metric:<20}  train {tr.mean():.3f}±{tr.std():.3f}"
              f"   test {te.mean():.3f}±{te.std():.3f}")
    print(f"{'─'*55}\n")

    # ── train final model on all data ─────────────────────────────────────────
    pipe.fit(X, y)
    joblib.dump(pipe, os.path.join(model_dir, "pipeline.joblib"))
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump({"feature_cols": feat_cols, "classes": list(le.classes_)}, f)
    print(f"  Model saved to {model_dir}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    y_pred = pipe.predict(X)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        le.inverse_transform(y), le.inverse_transform(y_pred),
        ax=ax_cm, colorbar=False,
    )
    ax_cm.set_title("Confusion matrix (train set)", fontsize=9)
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(model_dir, "confusion_matrix.pdf"),
                   bbox_inches="tight")

    # ── feature importances ───────────────────────────────────────────────────
    importances = pipe["clf"].feature_importances_
    order = np.argsort(importances)[::-1][:20]
    fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
    ax_fi.barh([feat_cols[i] for i in order[::-1]], importances[order[::-1]])
    ax_fi.set_xlabel("Importance")
    ax_fi.set_title("Top-20 feature importances", fontsize=9)
    fig_fi.tight_layout()
    fig_fi.savefig(os.path.join(model_dir, "feature_importances.pdf"),
                   bbox_inches="tight")

    # ── misclassified examples ────────────────────────────────────────────────
    wrong = df[(y_pred != y)].head(n_misclassified_examples)
    if not wrong.empty:
        trial_starts = np.sort(trial_table["start_time"].values)
        n = len(wrong)
        fig_mis, axes = plt.subplots(n, 2, figsize=(10, 3 * n),
                                     gridspec_kw={"width_ratios": [3, 1]})
        if n == 1:
            axes = [axes]
        for ax_row, (_, row) in zip(axes, wrong.iterrows()):
            plot_unit(ax_row[0], ax_row[1],
                      row["spike_times"], trial_starts,
                      waveform=row.get("waveform_mean"))
            pred_label = le.classes_[pipe.predict(row[feat_cols].values.reshape(1, -1))[0]]
            true_label = row["manual_label"]
            ax_row[0].set_title(
                f"True={true_label}  Pred={pred_label}  "
                + "  ".join(f"{c}={row[c]}" for c in LABEL_COLS),
                fontsize=7, loc="left", color="crimson",
            )
        fig_mis.suptitle("Misclassified examples", fontsize=9)
        fig_mis.tight_layout()
        fig_mis.savefig(os.path.join(model_dir, "misclassifications.pdf"),
                        bbox_inches="tight")

    plt.show()
    print(f"\n{classification_report(le.inverse_transform(y), le.inverse_transform(y_pred))}")
    return {"cv": cv_results, "pipeline": pipe, "feature_cols": feat_cols}
