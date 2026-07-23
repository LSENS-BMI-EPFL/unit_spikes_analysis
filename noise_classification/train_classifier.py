"""
train_classifier.py — train a calibrated Random Forest.

Versioning
----------
    <output_dir>/
    ├── labels.csv                              ← master
    ├── labels_20260521T134500.csv              ← snapshot at train time
    └── models/
        └── 20260521T134500_a1b2c3d4/
            ├── pipeline_calibrated.joblib
            ├── pipeline_raw.joblib
            ├── meta.json
            └── plots/

Usage
-----
    from train_classifier import train
    train(output_dir="noise_classification/", unit_table=..., trial_table=...)
"""

import hashlib, json, os, shutil
from datetime import datetime

import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from noise_classification._shared import LABEL_COLS, get_feature_cols, plot_unit


def _pipeline(calibrate: bool) -> Pipeline:
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                class_weight="balanced", random_state=0, n_jobs=-1)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", CalibratedClassifierCV(rf, cv=5, method="isotonic") if calibrate else rf),
    ])


def train(output_dir: str, unit_table: pd.DataFrame, trial_table: pd.DataFrame,
          n_splits: int = 5, n_misclassified: int = 6) -> dict:

    labels_csv = os.path.join(output_dir, "labels.csv")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"No labels found at {labels_csv}. Run the GUI first.")

    # ── versioning ─────────────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%dT%H%M%S")
    label_hash  = hashlib.md5(open(labels_csv, "rb").read()).hexdigest()[:8]
    mv          = f"{ts}_{label_hash}"            # model_version
    mdir        = os.path.join(output_dir, "models", mv)
    plots_dir   = os.path.join(mdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # snapshot labels
    snapshot = os.path.join(output_dir, f"labels_{ts}.csv")
    shutil.copy2(labels_csv, snapshot)
    print(f"  Label snapshot → {snapshot}")

    # ── data ───────────────────────────────────────────────────────────────────
    df   = pd.read_csv(labels_csv).merge(unit_table, on=LABEL_COLS, how="left")
    counts = df["manual_label"].value_counts().to_dict()
    print(f"  {len(df)} labels  {counts}  "
          f"({100*counts.get('NOISE',0)/len(df):.0f}% NOISE)")

    feat_cols = get_feature_cols(df)
    X  = df[feat_cols].values
    le = LabelEncoder()
    y  = le.fit_transform(df["manual_label"])

    # ── cross-validation (GroupKFold by mouse) ─────────────────────────────────
    n_mice    = df["mouse_id"].nunique()
    n_splits_ = min(n_splits, n_mice)
    cv        = GroupKFold(n_splits=n_splits_)
    pipe_cal  = _pipeline(calibrate=True)

    cv_res = cross_validate(
        pipe_cal, X, y, cv=cv, groups=df["mouse_id"].values,
        scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        return_train_score=True,
    )
    print(f"\n{'─'*55}  {n_splits_}-fold GroupKFold  [{mv}]")
    for m in ["accuracy", "f1_macro"]:
        tr, te = cv_res[f"train_{m}"], cv_res[f"test_{m}"]
        print(f"  {m:<18} train {tr.mean():.3f}±{tr.std():.3f}  "
              f"test {te.mean():.3f}±{te.std():.3f}")
    print(f"{'─'*55}\n")

    # ── fit final models ────────────────────────────────────────────────────────
    pipe_cal.fit(X, y)
    pipe_raw = _pipeline(calibrate=False); pipe_raw.fit(X, y)

    joblib.dump(pipe_cal, os.path.join(mdir, "pipeline_calibrated.joblib"))
    joblib.dump(pipe_raw, os.path.join(mdir, "pipeline_raw.joblib"))

    meta = {"model_version": mv, "label_snapshot": os.path.basename(snapshot),
            "label_hash": label_hash, "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_labeled": len(df), "class_counts": counts,
            "feature_cols": feat_cols, "classes": list(le.classes_),
            "n_mice": n_mice, "n_splits": n_splits_}
    json.dump(meta, open(os.path.join(mdir, "meta.json"), "w"), indent=2)
    print(f"  Model saved → {mdir}")

    # ── plots ───────────────────────────────────────────────────────────────────
    y_pred = pipe_cal.predict(X)
    y_prob = pipe_cal.predict_proba(X)
    ni     = list(le.classes_).index("NOISE")

    for fig, name in [
        (_cm_fig(le, y, y_pred, mv),                          "confusion_matrix.pdf"),
        (_cal_fig(y, y_prob[:, ni]),                           "calibration_curve.pdf"),
        (_fi_fig(pipe_raw, feat_cols),                         "feature_importances.pdf"),
        (_mis_fig(df[y_pred != y].head(n_misclassified),
                  le, pipe_cal, feat_cols,
                  np.sort(trial_table["start_time"].values)), "misclassifications.pdf"),
    ]:
        if fig is not None:
            fig.savefig(os.path.join(plots_dir, name), bbox_inches="tight")
            plt.close(fig)

    print(f"\n{classification_report(le.inverse_transform(y), le.inverse_transform(y_pred))}")
    return {"model_version": mv, "cv": cv_res, "pipeline_cal": pipe_cal,
            "pipeline_raw": pipe_raw, "feature_cols": feat_cols}


def _cm_fig(le, y, y_pred, title):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        le.inverse_transform(y), le.inverse_transform(y_pred), ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix\n{title}", fontsize=8); fig.tight_layout(); return fig

def _cal_fig(y, y_prob_noise):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    CalibrationDisplay.from_predictions(y, y_prob_noise, n_bins=10,
                                        ax=ax, name="RF calibrated")
    ax.set_title("Calibration (NOISE class)", fontsize=8); fig.tight_layout(); return fig

def _fi_fig(pipe_raw, feat_cols):
    imp = pipe_raw["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([feat_cols[i] for i in order[::-1]], imp[order[::-1]])
    ax.set_xlabel("Importance"); ax.set_title("Feature importances", fontsize=9)
    fig.tight_layout(); return fig

def _mis_fig(wrong, le, pipe_cal, feat_cols, trial_starts):
    if wrong.empty: return None
    n = len(wrong)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3*n),
                              gridspec_kw={"width_ratios": [3, 1]})
    if n == 1: axes = [axes]
    for ax_row, (_, row) in zip(axes, wrong.iterrows()):
        plot_unit(ax_row[0], ax_row[1], row["spike_times"], trial_starts,
                  waveform=row.get("waveform_mean"))
        pred = le.classes_[pipe_cal.predict(row[feat_cols].values.reshape(1,-1))[0]]
        ax_row[0].set_title(f"True={row['manual_label']}  Pred={pred}",
                             fontsize=7, loc="left", color="crimson")
    fig.suptitle("Misclassified examples", fontsize=9); fig.tight_layout(); return fig
