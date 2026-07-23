"""
apply_classifier.py — apply trained model to unseen units.

Predictions are versioned per model:
    <output_dir>/predictions/predictions_{model_version}.csv

If predictions already exist for a given model version they are returned
immediately without recomputation.

Usage
-----
    from apply_classifier import apply, get_uncertain_units

    preds    = apply(unit_table, output_dir="noise_classification/")
    uncertain = get_uncertain_units(unit_table, output_dir="noise_classification/", n=200)
"""

import json, os
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from noise_classification._shared import LABEL_COLS


def _load(output_dir: str, model_version: str | None):
    mdir_root = os.path.join(output_dir, "models")
    if model_version is None:
        versions = sorted(os.listdir(mdir_root))
        if not versions: raise FileNotFoundError(f"No models in {mdir_root}")
        model_version = versions[-1]
        print(f"  Using latest model: {model_version}")
    mdir  = os.path.join(mdir_root, model_version)
    meta  = json.load(open(os.path.join(mdir, "meta.json")))
    pipe_cal = joblib.load(os.path.join(mdir, "pipeline_calibrated.joblib"))
    pipe_raw = joblib.load(os.path.join(mdir, "pipeline_raw.joblib"))
    classes  = np.array(meta["classes"])
    return pipe_cal, pipe_raw, meta, meta["feature_cols"], classes, model_version


def apply(
    unit_table: pd.DataFrame,
    output_dir: str = "noise_classification/",
    model_version: str | None = None,
    bc_labels: tuple = ("good", "mua"),
    shap_summary: bool = True,
) -> pd.DataFrame:

    pipe_cal, pipe_raw, meta, feat_cols, classes, mv = _load(output_dir, model_version)
    noise_idx = list(classes).index("NOISE")

    out_csv = os.path.join(output_dir, "predictions", f"predictions_{mv}.csv")
    if os.path.exists(out_csv):
        print(f"  Predictions already exist for {mv} → loading {out_csv}")
        return pd.read_csv(out_csv)

    df = unit_table[unit_table["bc_label"].isin(bc_labels)].copy()
    print(f"  Model {mv}  |  screening {len(df):,} units …")

    # fill missing feature columns
    for c in [c for c in feat_cols if c not in df.columns]:
        df[c] = np.nan
    X = df[feat_cols].values

    df["pred_label"]      = classes[pipe_cal.predict(X)]
    df["pred_noise_prob"] = pipe_cal.predict_proba(X)[:, noise_idx]
    df["model_version"]   = mv

    n_noise = (df["pred_label"] == "NOISE").sum()
    print(f"  Predicted NOISE: {n_noise:,} / {len(df):,} ({100*n_noise/len(df):.1f}%)")

    if shap_summary:
        _shap(df, pipe_raw, feat_cols, output_dir, mv)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = LABEL_COLS + ["bc_label", "pred_label", "pred_noise_prob", "model_version"]
    df[cols].to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")
    return df


def get_uncertain_units(
    unit_table: pd.DataFrame,
    output_dir: str = "noise_classification/",
    model_version: str | None = None,
    n: int = 200,
    bc_labels: tuple = ("good", "mua"),
) -> pd.DataFrame:
    """Return N units with pred_noise_prob closest to 0.5 (most uncertain)."""
    pipe_cal, _, meta, feat_cols, classes, mv = _load(output_dir, model_version)
    noise_idx = list(classes).index("NOISE")

    df = unit_table[unit_table["bc_label"].isin(bc_labels)].copy()
    for c in [c for c in feat_cols if c not in df.columns]:
        df[c] = np.nan
    df["pred_noise_prob"] = pipe_cal.predict_proba(df[feat_cols].values)[:, noise_idx]

    result = (df.assign(_u=lambda d: (d["pred_noise_prob"] - 0.5).abs())
                .sort_values("_u").head(n).drop(columns="_u").reset_index(drop=True))
    print(f"  Uncertain units: {len(result)}  prob range "
          f"{result['pred_noise_prob'].min():.2f}–{result['pred_noise_prob'].max():.2f}")
    return result


def adjust_for_prior(
    prob: pd.Series | np.ndarray,
    p_noise_train: float,
    p_noise_true: float,
) -> np.ndarray:
    """
    Correct pred_noise_prob for the mismatch between the training class
    distribution and the true prevalence in the full dataset.

    Because labeled units are sampled from the most suspicious candidates,
    NOISE is heavily overrepresented in training relative to the real 100k
    dataset.  This inflates all predicted probabilities.  The correction uses
    the odds form of Bayes' theorem to rescale to the true prior.

    Parameters
    ----------
    prob          : raw model output (pred_noise_prob column).
    p_noise_train : fraction of NOISE in your labeled set
                    (read directly from labels.csv).
    p_noise_true  : your best estimate of true prevalence in the full dataset
                    (e.g. 0.01 for ~1%).  A rough estimate is fine — the
                    correction is not sensitive to small errors here.

    Returns
    -------
    np.ndarray of corrected probabilities in [0, 1].
    """
    p = np.asarray(prob, dtype=float)
    odds          = p / (1 - p + 1e-9)
    prior_ratio   = (p_noise_true / (1 - p_noise_true)) / \
                    (p_noise_train / (1 - p_noise_train))
    odds_corrected = odds * prior_ratio
    return odds_corrected / (1 + odds_corrected)


def get_top_k_for_review(
    predictions: pd.DataFrame,
    unit_table: pd.DataFrame,
    p_noise_train: float,
    p_noise_true: float,
    k: int = 2000,
) -> pd.DataFrame:
    """
    Return the top-k most likely noise units from the full dataset,
    corrected for the training/test prior mismatch, ready to pass
    directly to run_labeling_gui(pre_ranked_candidates=...).

    Parameters
    ----------
    predictions   : output of apply() — needs LABEL_COLS + pred_noise_prob.
    unit_table    : full unit table (needs spike_times, waveform_mean, bc_label).
    p_noise_train : fraction of NOISE in your labeled set.
    p_noise_true  : estimated true NOISE prevalence in the full dataset.
    k             : how many units to return for manual review.

    Returns
    -------
    pd.DataFrame sorted by pred_noise_prob_corrected descending, with
    spike_times and waveform_mean joined in so the GUI can plot them.
    """
    df = predictions.copy()
    df["pred_noise_prob_corrected"] = adjust_for_prior(
        df["pred_noise_prob"], p_noise_train, p_noise_true
    )

    # join spike_times / waveform_mean back in for the GUI
    cols_from_unit = [c for c in ["spike_times", "waveform_mean", "bc_label"]
                      if c in unit_table.columns]
    df = df.merge(unit_table[LABEL_COLS + cols_from_unit], on=LABEL_COLS, how="left",
                  suffixes=("", "_unit"))
    # prefer unit_table bc_label if both present
    if "bc_label_unit" in df.columns:
        df["bc_label"] = df["bc_label_unit"]
        df = df.drop(columns=["bc_label_unit"])

    top_k = (df.sort_values("pred_noise_prob_corrected", ascending=False)
               .head(k)
               .reset_index(drop=True))

    print(f"  Top-{k} for review — corrected prob range "
          f"{top_k['pred_noise_prob_corrected'].min():.3f}–"
          f"{top_k['pred_noise_prob_corrected'].max():.3f}  "
          f"(raw: {top_k['pred_noise_prob'].min():.3f}–"
          f"{top_k['pred_noise_prob'].max():.3f})")
    return top_k


def _shap(df, pipe_raw, feat_cols, output_dir, mv):
    noise_rows = df[df["pred_label"] == "NOISE"]
    if noise_rows.empty: return
    X_imp = pipe_raw["imputer"].transform(noise_rows[feat_cols].values)
    rf    = pipe_raw["clf"]
    sv    = shap.TreeExplainer(rf).shap_values(X_imp)
    ni    = list(rf.classes_).index(1)
    sv    = sv[ni] if isinstance(sv, list) else sv
    shap.summary_plot(sv, X_imp, feature_names=feat_cols, show=False)
    plt.title(f"SHAP — NOISE units (n={len(noise_rows)})", fontsize=9)
    out = os.path.join(output_dir, "models", mv, "plots", "shap_summary.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  SHAP → {out}")
