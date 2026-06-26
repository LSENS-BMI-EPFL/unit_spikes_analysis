"""
sigmoid_modulation.py
---------------------
For each good neuron, fits three models (constant, linear, sigmoid) to
trial-by-trial firing rates (baseline and sensory-evoked, baseline-corrected),
for whisker-active and auditory-active trials separately.

Sigmoid validity constraints:
  - t0 within [t[0]+BORDER_MARGIN, t[-1]-BORDER_MARGIN]
  - S-shaped plateau: |k| * (t[-1]-t[0]) > S_SHAPE_MIN_SPAN

Outputs
-------
- Per-session CSV
- Per-mouse figures (examples, fraction modulated, lag distributions)
- Summary figures across dataset (R+ vs R-)

Modes
-----
run_analysis()      — full pipeline (analysis + figures)
run_figures_only()  — load existing CSVs, regenerate all figures
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import expit
from scipy.stats import mannwhitneyu, gaussian_kde
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from allen_utils import get_custom_area_order

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_OUT = Path(r"M:\analysis\Axel_Bisi\combined_results\sigmoid_analysis")
BASE_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT = BASE_OUT / "figures"
FIG_OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_START   = -2.0    # s before stim
BASELINE_END     = -0.005  # s before stim
EVOKED_START     =  0.005  # s after stim
EVOKED_END       =  0.050  # s after stim

AIC_DELTA_THRESH = 2.0     # sigmoid ΔAIC advantage required
BORDER_MARGIN    = 5       # trials: t0 must be in [margin, n-1-margin]
S_SHAPE_MIN_SPAN = 4.0     # |k|*(t[-1]-t[0]) > this → both tails in plateau
N_EXAMPLE        = 10      # example neurons per context
N_JOBS           = -1

CATS      = ["constant", "linear", "sigmoid_down", "sigmoid_up"]
CAT_COLORS = ["#aaaaaa", "#f0a500", "#4477aa", "#cc3333"]
RG_COLORS  = {1: "#228B22", 0: "#DC143C"}

# ---------------------------------------------------------------------------
# Firing rate helpers (vectorized)
# ---------------------------------------------------------------------------

def compute_fr_vectorized(spike_times: np.ndarray, trial_starts: np.ndarray,
                           t_start: float, t_end: float) -> np.ndarray:
    """Vectorized spike counting via searchsorted. spike_times must be sorted."""
    dur    = t_end - t_start
    lo_idx = np.searchsorted(spike_times, trial_starts + t_start, side="left")
    hi_idx = np.searchsorted(spike_times, trial_starts + t_end,   side="left")
    return (hi_idx - lo_idx).astype(float) / dur


def compute_epoch_fr(spike_times: np.ndarray, trial_starts: np.ndarray):
    """Returns baseline_fr, evoked_bc (Hz, per trial). spike_times must be sorted."""
    baseline = compute_fr_vectorized(spike_times, trial_starts, BASELINE_START, BASELINE_END)
    evoked   = compute_fr_vectorized(spike_times, trial_starts, EVOKED_START,   EVOKED_END)
    return baseline, evoked - baseline


# ---------------------------------------------------------------------------
# Trial selection
# ---------------------------------------------------------------------------

def get_active_trial_starts(trial_table: pd.DataFrame, trial_type: str):
    """Returns (trial_starts, n_trials) for active-context trials."""
    mask = (
        (trial_table["trial_type"] == f"{trial_type}_trial") &
        (trial_table["context"]    != "passive")
    )
    return trial_table.loc[mask, "start_time"].values, int(mask.sum())


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _aic(n: int, rss: float, k: int) -> float:
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def _fit_constant(y: np.ndarray):
    mu  = y.mean()
    rss = float(np.sum((y - mu) ** 2))
    return mu, rss, _aic(len(y), rss, 1)


def _fit_linear(t: np.ndarray, y: np.ndarray):
    tm, ym = t.mean(), y.mean()
    slope  = np.sum((t - tm) * (y - ym)) / np.sum((t - tm) ** 2)
    ic     = ym - slope * tm
    rss    = float(np.sum((y - slope * t - ic) ** 2))
    return slope, ic, rss, _aic(len(t), rss, 2)


def _sigmoid_fn(t, L, k, t0, b):
    return L * expit(k * (t - t0)) + b


def _fit_sigmoid(t: np.ndarray, y: np.ndarray):
    """
    Fit sigmoid with:
      - t0 bounded to [t[0]+BORDER_MARGIN, t[-1]-BORDER_MARGIN]
      - S-shape check: |k| * trial_range > S_SHAPE_MIN_SPAN
    Returns (L, k, t0, b, rss, aic, valid).
    """
    t_range = float(t[-1] - t[0])
    t_lo    = float(t[0]  + BORDER_MARGIN)
    t_hi    = float(t[-1] - BORDER_MARGIN)

    if t_lo >= t_hi:
        return np.nan, np.nan, np.nan, np.nan, np.inf, np.inf, False

    n  = len(t)
    p0 = [float(np.ptp(y)), 0.1, float(t[n // 2]), float(y.min())]
    bounds = ([-np.inf, -5.0, t_lo, -np.inf],
              [ np.inf,  5.0, t_hi,  np.inf])
    try:
        popt, _ = curve_fit(_sigmoid_fn, t, y, p0=p0, bounds=bounds, maxfev=10_000)
        L, k, t0, b = popt
        yhat     = _sigmoid_fn(t, *popt)
        rss      = float(np.sum((y - yhat) ** 2))
        aic      = _aic(n, rss, 4)
        s_shaped = abs(k) * t_range > S_SHAPE_MIN_SPAN
        return L, k, t0, b, rss, aic, s_shaped
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.inf, np.inf, False


def model_comparison(t: np.ndarray, y: np.ndarray) -> dict:
    mu,    rss_c, aic_c              = _fit_constant(y)
    sl, ic, rss_l, aic_l             = _fit_linear(t, y)
    L, k, t0, b, rss_s, aic_s, valid = _fit_sigmoid(t, y)

    sigmoid_wins = (
        valid and
        aic_s < aic_c and
        aic_s < aic_l and
        (aic_s - min(aic_c, aic_l)) < -AIC_DELTA_THRESH
    )

    if sigmoid_wins:
        t0_idx    = int(np.round(t0))
        direction = 1 if L > 0 else -1
    else:
        t0 = t0_idx = direction = np.nan

    return {
        "aic_constant": aic_c, "aic_linear": aic_l, "aic_sigmoid": aic_s,
        "rss_constant": rss_c, "rss_linear": rss_l, "rss_sigmoid": rss_s,
        "best_model":   "sigmoid" if sigmoid_wins else ("constant" if aic_c <= aic_l else "linear"),
        "sigmoid_wins": sigmoid_wins,
        "L": L, "k": k, "t0": t0, "b": b,
        "t0_idx": t0_idx, "direction": direction,
    }


# ---------------------------------------------------------------------------
# Per-neuron worker (parallelized)
# ---------------------------------------------------------------------------

def _neuron_worker(unit_id, area, mouse_id, session_id, probe, reward_group,
                   spike_times, trial_starts_w, n_w, trial_starts_a, n_a):
    spk  = np.sort(np.asarray(spike_times))
    meta = dict(unit_id=unit_id, area_acronym_custom=area, mouse_id=mouse_id,
                session_id=session_id, probe=probe, reward_group=reward_group)
    results = []
    for trial_type, trial_starts, n_trials in [
        ("whisker",  trial_starts_w, n_w),
        ("auditory", trial_starts_a, n_a),
    ]:
        if n_trials < 10:
            continue
        baseline, evoked_bc = compute_epoch_fr(spk, trial_starts)
        t = np.arange(n_trials, dtype=float)
        for epoch, y in [("baseline", baseline), ("evoked", evoked_bc)]:
            res = model_comparison(t, y)
            res.update(meta)
            res.update({"trial_type": trial_type, "epoch": epoch,
                        "n_trials": n_trials, "mean_fr": float(y.mean()),
                        "fr_series": y.tolist()})
            results.append(res)
    return results


def analyse_session(unit_table: pd.DataFrame, trial_table: pd.DataFrame) -> list[dict]:
    """Parallel analysis of all good neurons in one session."""
    good = unit_table[unit_table["bc_label"] == "good"]
    if good.empty:
        return []
    trial_starts_w, n_w = get_active_trial_starts(trial_table, "whisker")
    trial_starts_a, n_a = get_active_trial_starts(trial_table, "auditory")
    rows    = good.to_dict("records")
    batches = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_neuron_worker)(
            r["unit_id"], r["area_acronym_custom"], r["mouse_id"], r["session_id"],
            r.get("probe", np.nan), r.get("reward_group", np.nan),
            r["spike_times"], trial_starts_w, n_w, trial_starts_a, n_a,
        )
        for r in rows
    )
    return [item for batch in batches for item in batch]


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def save_results(results: list[dict], session_id: str) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.drop(columns=["fr_series"], errors="ignore").to_csv(
        BASE_OUT / f"{session_id}_sigmoid_results.csv", index=False
    )
    return df


def load_all_results() -> pd.DataFrame:
    """Load all per-session CSVs from BASE_OUT. Warns for missing files."""
    csvs = sorted(BASE_OUT.glob("*_sigmoid_results.csv"))
    if not csvs:
        raise FileNotFoundError(f"No sigmoid result CSVs found in {BASE_OUT}")
    dfs = []
    for p in csvs:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"WARNING: could not read {p.name}: {e}")
    if not dfs:
        raise RuntimeError("No result files could be loaded.")
    return pd.concat(dfs, ignore_index=True)


def load_lag_csv() -> pd.DataFrame:
    p = BASE_OUT / "lag_analysis.csv"
    if not p.exists():
        raise FileNotFoundError(f"lag_analysis.csv not found in {BASE_OUT}")
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _mouse_fig_dir(mouse_id: str) -> Path:
    d = FIG_OUT / mouse_id
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Figures — example neurons
# ---------------------------------------------------------------------------

def _plot_example_neuron(row: pd.Series, out_path: Path):
    y      = np.asarray(row["fr_series"])
    t      = np.arange(len(y), dtype=float)
    t_fine = np.linspace(t[0], t[-1], 300)

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, y, "o", ms=3, color="gray", alpha=0.6, label="FR data")
    ax1.axhline(y.mean(), ls="--", color="blue", lw=1.5, label="Constant")
    sl, ic, *_ = _fit_linear(t, y)
    ax1.plot(t_fine, sl * t_fine + ic, "--", color="orange", lw=1.5, label="Linear")
    ax1.plot(t_fine, _sigmoid_fn(t_fine, row["L"], row["k"], row["t0"], row["b"]),
             "-", color="red", lw=2, label="Sigmoid")
    ax1.axvline(row["t0"], color="red", ls=":", lw=1, label=f"t0={row['t0']:.1f}")
    ax1.set_xlabel("Active trial index"); ax1.set_ylabel("FR (Hz)")
    ax1.set_title(f"{row['unit_id']} | {row['area_acronym_custom']} | {row['trial_type']} {row['epoch']}")
    ax1.legend(fontsize=7)

    ax2    = fig.add_subplot(gs[0, 2])
    models = ["constant", "linear", "sigmoid"]
    aics   = [row[f"aic_{m}"] for m in models]
    bars   = ax2.bar(models, aics, color=["blue", "orange", "red"], alpha=0.7)
    min_a  = min(aics)
    for bar, aic in zip(bars, aics):
        ax2.text(bar.get_x() + bar.get_width() / 2, aic + 0.5,
                 f"{aic - min_a:+.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("AIC"); ax2.set_title("Model AIC")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, y - _sigmoid_fn(t, row["L"], row["k"], row["t0"], row["b"]),
             "o", ms=3, color="gray", alpha=0.6)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xlabel("Trial"); ax3.set_ylabel("Residual (Hz)"); ax3.set_title("Sigmoid residuals")

    ax4 = fig.add_subplot(gs[1, 1]); ax4.axis("off")
    ax4.text(0.05, 0.95, (
        f"L  = {row['L']:.3f}\n"
        f"k  = {row['k']:.4f}\n"
        f"t0 = {row['t0']:.2f}\n"
        f"b  = {row['b']:.3f}\n"
        f"t0_idx    = {row['t0_idx']}\n"
        f"direction = {'↑' if row['direction']==1 else '↓'}\n"
        f"|k|*range = {abs(row['k'])*(len(y)-1):.2f} (>{S_SHAPE_MIN_SPAN})\n"
        f"ΔAIC(s-l) = {row['aic_sigmoid']-row['aic_linear']:.2f}\n"
        f"ΔAIC(s-c) = {row['aic_sigmoid']-row['aic_constant']:.2f}"
    ), transform=ax4.transAxes, va="top", fontsize=9, family="monospace")
    ax4.set_title("Parameters")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(["vs Linear", "vs Constant"],
            [row["aic_sigmoid"] - row["aic_linear"],
             row["aic_sigmoid"] - row["aic_constant"]],
            color=["orange", "blue"], alpha=0.7)
    ax5.axhline(-AIC_DELTA_THRESH, color="k", ls="--", lw=1,
                label=f"threshold={-AIC_DELTA_THRESH}")
    ax5.set_ylabel("ΔAIC (sigmoid − other)"); ax5.set_title("Sigmoid advantage")
    ax5.legend(fontsize=7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example_neurons(results_df: pd.DataFrame, n: int = N_EXAMPLE,
                          tag: str = "", out_dir: Path = FIG_OUT):
    """Top-n sigmoid neurons by |L|. Requires fr_series column (in-memory only)."""
    if "fr_series" not in results_df.columns:
        return
    sig = results_df[results_df["sigmoid_wins"]].copy()
    if sig.empty:
        return
    examples = sig.loc[sig["L"].abs().nlargest(n).index]
    for _, row in examples.iterrows():
        uid  = str(row["unit_id"]).replace("/", "_")
        _plot_example_neuron(row, out_dir / f"example_{tag}_{uid}.png")


# ---------------------------------------------------------------------------
# Figures — fraction modulated
# ---------------------------------------------------------------------------

def _classify_category(df: pd.DataFrame) -> pd.Series:
    cat = df["best_model"].copy()
    sig = df["sigmoid_wins"]
    cat[sig & (df["direction"] ==  1)] = "sigmoid_up"
    cat[sig & (df["direction"] == -1)] = "sigmoid_down"
    return cat


def _stacked_bar(mean_df: pd.DataFrame, areas: list, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(areas) * 1.2), 5))
    x      = np.arange(len(areas))
    bottom = np.zeros(len(areas))
    for cat, color in zip(CATS, CAT_COLORS):
        vals = mean_df.reindex(areas)[cat].fillna(0).values
        ax.bar(x, vals, bottom=bottom, color=color, label=cat, alpha=0.85)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(areas, rotation=45, ha="right")
    ax.set_ylabel("Fraction of neurons"); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fraction_modulated(results_df: pd.DataFrame, trial_type: str, epoch: str,
                             out_dir: Path = FIG_OUT, title_prefix: str = ""):
    sub = results_df[
        (results_df["trial_type"] == trial_type) &
        (results_df["epoch"]     == epoch)
    ].copy()
    if sub.empty:
        return
    sub["category"] = _classify_category(sub)
    ct      = pd.crosstab([sub["mouse_id"], sub["area_acronym_custom"]], sub["category"])
    ct      = ct.reindex(columns=CATS, fill_value=0)
    ct      = ct.div(ct.sum(axis=1), axis=0)
    mean_df = ct.groupby(level="area_acronym_custom").mean().reindex(columns=CATS, fill_value=0)
    areas   = sorted(mean_df.index)
    _stacked_bar(mean_df, areas,
                 f"{title_prefix}Modulation fractions — {trial_type} {epoch}",
                 out_dir / f"fraction_modulated_{trial_type}_{epoch}.png")


# ---------------------------------------------------------------------------
# Lag helpers
# ---------------------------------------------------------------------------

def compute_lags(results_df: pd.DataFrame, learning_df: pd.DataFrame) -> pd.DataFrame:
    """whisker + evoked + sigmoid_wins only. lag = t0_idx − learning_trial."""
    sub = results_df[
        (results_df["trial_type"]    == "whisker") &
        (results_df["epoch"]        == "evoked")  &
        (results_df["sigmoid_wins"] == True)
    ].copy()
    merged = sub.merge(learning_df, on="mouse_id", how="left")
    merged["lag"] = merged["t0_idx"] - merged["learning_trial"]
    return merged


def get_shared_areas(lag_df: pd.DataFrame, min_mice: int = 3) -> list[str]:
    counts = (
        lag_df.groupby(["area_acronym_custom", "reward_group"])["mouse_id"]
        .nunique().unstack(fill_value=0)
    )
    print(counts)
    mask = (counts.get(1, pd.Series(0, index=counts.index)) >= min_mice) & \
           (counts.get(0, pd.Series(0, index=counts.index)) >= min_mice)
    return sorted(counts.index[mask].tolist())


def _ordered_areas(areas: list[str]) -> list[str]:
    """Return areas sorted by Allen custom order; unknown areas appended at end."""
    allen_order = get_custom_area_order()
    known   = [a for a in allen_order if a in areas]
    unknown = [a for a in areas if a not in allen_order]
    return known + sorted(unknown)


# ---------------------------------------------------------------------------
# Figures — per-area lag distributions (existing violin style)
# ---------------------------------------------------------------------------

def _violin_lag_panel(ax, rp_vals, rm_vals, area):
    for pos, vals, color in [(0, rp_vals, RG_COLORS[1]),
                              (1, rm_vals, RG_COLORS[0])]:
        if len(vals) == 0:
            continue
        parts = ax.violinplot([vals], positions=[pos], showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.5)
        ax.scatter(np.random.normal(pos, 0.05, size=len(vals)),
                   vals, color=color, s=15, alpha=0.7, zorder=3)
        ax.axhline(np.median(vals), color=color, ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R-"])
    ax.set_xlabel("Reward group"); ax.set_title(area)


def plot_lag_distribution(lag_df: pd.DataFrame, direction: int,
                           areas_shared: list[str], tag: str = "",
                           out_dir: Path = FIG_OUT):
    dir_label = "increased" if direction == 1 else "decreased"
    sub = lag_df[
        (lag_df["direction"] == direction) &
        (lag_df["area_acronym_custom"].isin(areas_shared))
    ].dropna(subset=["lag"])
    if sub.empty or not areas_shared:
        return

    mouse_df = sub.groupby(["mouse_id", "area_acronym_custom", "reward_group"])["lag"].median().reset_index()

    for level, df in [("mouse", mouse_df), ("neuron", sub)]:
        fig, axes = plt.subplots(1, len(areas_shared),
                                 figsize=(3.5 * len(areas_shared), 5), sharey=True)
        axes = [axes] if len(areas_shared) == 1 else list(axes)
        pvals = {}
        for ax, area in zip(axes, areas_shared):
            rp = df[(df["area_acronym_custom"] == area) & (df["reward_group"] == 1)]["lag"].dropna().values
            rm = df[(df["area_acronym_custom"] == area) & (df["reward_group"] == 0)]["lag"].dropna().values
            _violin_lag_panel(ax, rp, rm, area)
            if len(rp) >= 3 and len(rm) >= 3:
                _, p = mannwhitneyu(rp, rm, alternative="two-sided")
                pvals[area] = p

        areas_tested = [a for a in pvals if not np.isnan(pvals[a])]
        if areas_tested:
            _, pvals_corr, _, _ = multipletests(
                [pvals[a] for a in areas_tested], method="fdr_bh"
            )
            for area, pc in zip(areas_tested, pvals_corr):
                axes[areas_shared.index(area)].set_title(f"{area}\np_fdr={pc:.3f}")

        axes[0].set_ylabel(f"Lag (neural − behavioral trial), {level} level")
        fig.suptitle(f"Lag — {dir_label} neurons ({level} level) {tag}", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"lag_{dir_label}_{level}_{tag}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_lag_difference_distribution(lag_df: pd.DataFrame, direction: int,
                                      tag: str = "", out_dir: Path = FIG_OUT):
    dir_label = "increased" if direction == 1 else "decreased"
    sub = lag_df[lag_df["direction"] == direction].dropna(subset=["lag"])
    if sub.empty:
        return
    mouse_lags = sub.groupby("mouse_id")["lag"].median()
    fig, axes  = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in [
        (axes[0], sub["lag"].values,  "neuron"),
        (axes[1], mouse_lags.values,  "mouse"),
    ]:
        ax.hist(data, bins=30, color="#888888", alpha=0.6, density=True)
        ax.axvline(data.mean(),     color="red",  lw=2, ls="--",
                   label=f"mean={data.mean():.1f}")
        ax.axvline(np.median(data), color="blue", lw=2, ls="-",
                   label=f"median={np.median(data):.1f}")
        ax.axvline(0, color="k", lw=1, ls=":")
        ax.set_xlabel("Lag (neural − behavioral trial)")
        ax.set_ylabel("Density")
        ax.set_title(f"{dir_label} — {label} level (n={len(data)})")
        ax.legend(fontsize=8)
    fig.suptitle(f"Lag distribution — {dir_label} neurons {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"lag_distribution_{dir_label}_{tag}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — summary pointplot (areas × lag, Allen order, R+ vs R-)
# ---------------------------------------------------------------------------

def plot_lag_summary_pointplot(lag_df: pd.DataFrame, shared_areas: list[str],
                                out_dir: Path = FIG_OUT):
    """
    Two-panel figure (increased / decreased), side by side.
    Y-axis: areas in Allen order.
    X-axis: lag (trials).
    Points: per-mouse medians, colored by R+/R-, jittered between groups.
    Error bar: SEM across mice per area per group.
    """
    ordered = _ordered_areas(shared_areas)
    if not ordered:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(ordered) * 0.55 + 1)),
                              sharey=True)

    for ax, direction in zip(axes, [1, -1]):
        dir_label = "Increased" if direction == 1 else "Decreased"
        sub = lag_df[
            (lag_df["direction"] == direction) &
            (lag_df["area_acronym_custom"].isin(ordered))
        ].dropna(subset=["lag"])

        mouse_med = (sub.groupby(["mouse_id", "area_acronym_custom", "reward_group"])["lag"]
                     .median().reset_index())

        pvals = {}
        for i, area in enumerate(ordered):
            for jitter, rg in [(-0.15, 1), (0.15, 0)]:
                vals = mouse_med[
                    (mouse_med["area_acronym_custom"] == area) &
                    (mouse_med["reward_group"] == rg)
                ]["lag"].dropna().values
                if len(vals) == 0:
                    continue
                y_pos  = i + jitter
                color  = RG_COLORS[rg]
                mean_v = vals.mean()
                sem_v  = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                ax.scatter(vals, np.full(len(vals), y_pos),
                           color=color, s=20, alpha=0.5, zorder=3)
                ax.errorbar(mean_v, y_pos, xerr=sem_v,
                            fmt="o", color=color, ms=7, lw=2, capsize=3, zorder=4)

            # Mann-Whitney
            rp_v = mouse_med[(mouse_med["area_acronym_custom"] == area) &
                              (mouse_med["reward_group"] == 1)]["lag"].dropna().values
            rm_v = mouse_med[(mouse_med["area_acronym_custom"] == area) &
                              (mouse_med["reward_group"] == 0)]["lag"].dropna().values
            if len(rp_v) >= 3 and len(rm_v) >= 3:
                _, p = mannwhitneyu(rp_v, rm_v, alternative="two-sided")
                pvals[area] = p

        # FDR-BH
        areas_tested = [a for a in pvals]
        if areas_tested:
            _, pvals_corr, _, _ = multipletests(
                [pvals[a] for a in areas_tested], method="fdr_bh"
            )
            for area, pc in zip(areas_tested, pvals_corr):
                if pc < 0.05:
                    i = ordered.index(area)
                    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 5,
                            i, f"* p={pc:.3f}", va="center", fontsize=7, color="k")

        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered, fontsize=8)
        ax.set_xlabel("Lag (neural − behavioral trial)")
        ax.set_title(dir_label)
        ax.invert_yaxis()

    # shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=RG_COLORS[rg], ms=8, label=rg)
               for rg in [1, 0]]
    axes[1].legend(handles=handles, loc="lower right", fontsize=9)

    fig.suptitle("Lag by area — mouse-median pointplot (SEM)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "lag_summary_pointplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — summary heatmap (areas × lag bins, neuron-level density)
# ---------------------------------------------------------------------------

def plot_lag_summary_heatmap(lag_df: pd.DataFrame, shared_areas: list[str],
                              out_dir: Path = FIG_OUT,
                              lag_range: tuple = (-40, 40), n_bins: int = 80):
    """
    Two figures (increased / decreased), each with two heatmap panels (R+ / R-).
    Y-axis: areas in Allen order.
    X-axis: lag bins.
    Color: KDE density across all neurons for that area × group.
    """
    ordered = _ordered_areas(shared_areas)
    if not ordered:
        return

    bins    = np.linspace(lag_range[0], lag_range[1], n_bins + 1)
    bin_ctr = 0.5 * (bins[:-1] + bins[1:])

    for direction in [1, -1]:
        dir_label = "increased" if direction == 1 else "decreased"
        sub = lag_df[
            (lag_df["direction"] == direction) &
            (lag_df["area_acronym_custom"].isin(ordered))
        ].dropna(subset=["lag"])
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(ordered) * 0.55 + 1)),
                                  sharey=True)

        for ax, rg in zip(axes, [1, 0]):
            mat = np.zeros((len(ordered), n_bins))
            for i, area in enumerate(ordered):
                vals = sub[(sub["area_acronym_custom"] == area) &
                           (sub["reward_group"] == rg)]["lag"].dropna().values
                if len(vals) < 2:
                    continue
                # clip to range, then KDE
                vals_clip = vals[(vals >= lag_range[0]) & (vals <= lag_range[1])]
                if len(vals_clip) < 2:
                    continue
                try:
                    kde  = gaussian_kde(vals_clip, bw_method="scott")
                    row  = kde(bin_ctr)
                    mat[i] = row / row.max() if row.max() > 0 else row
                except Exception:
                    pass

            im = ax.imshow(mat, aspect="auto", origin="upper",
                           extent=[lag_range[0], lag_range[1], len(ordered) - 0.5, -0.5],
                           cmap="YlOrRd", vmin=0, vmax=1)
            ax.axvline(0, color="w", lw=1, ls="--")
            ax.set_yticks(range(len(ordered)))
            ax.set_yticklabels(ordered, fontsize=8)
            ax.set_xlabel("Lag (neural − behavioral trial)")
            ax.set_title(f"{rg} (n neurons per area shown)")
            plt.colorbar(im, ax=ax, label="Norm. density")

        fig.suptitle(f"Lag density heatmap — {dir_label} neurons", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"lag_summary_heatmap_{dir_label}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-mouse figures
# ---------------------------------------------------------------------------

def make_mouse_figures(mouse_id: str, results_df: pd.DataFrame,
                        lag_df: pd.DataFrame):
    mdf   = results_df[results_df["mouse_id"] == mouse_id]
    m_lag = lag_df[lag_df["mouse_id"] == mouse_id]
    out   = _mouse_fig_dir(mouse_id)

    for stim, tag in [("whisker", "whisker_evoked"), ("auditory", "auditory_evoked")]:
        plot_example_neurons(
            mdf[(mdf["trial_type"] == stim) & (mdf["epoch"] == "evoked")],
            tag=tag, out_dir=out
        )

    for stim in ["whisker", "auditory"]:
        for epoch in ["baseline", "evoked"]:
            plot_fraction_modulated(mdf, stim, epoch,
                                    out_dir=out, title_prefix=f"{mouse_id} — ")

    if m_lag.empty:
        return
    areas_mouse = _ordered_areas(sorted(m_lag["area_acronym_custom"].dropna().unique().tolist()))
    for direction in [1, -1]:
        plot_lag_distribution(m_lag, direction, areas_mouse,
                               tag=mouse_id, out_dir=out)
        plot_lag_difference_distribution(m_lag, direction,
                                          tag=mouse_id, out_dir=out)


# ---------------------------------------------------------------------------
# Summary figures (shared logic used by both run modes)
# ---------------------------------------------------------------------------

def _make_summary_figures(results_df: pd.DataFrame, lag_df: pd.DataFrame):
    print("Summary figures ...")

    # fraction modulated
    for stim in ["whisker", "auditory"]:
        for epoch in ["baseline", "evoked"]:
            plot_fraction_modulated(results_df, stim, epoch,
                                    out_dir=FIG_OUT, title_prefix="Summary — ")

    shared_areas = get_shared_areas(lag_df)
    print(f"Shared areas (R+/R- ≥3 mice): {shared_areas}")

    for direction in [1, -1]:
        plot_lag_distribution(lag_df, direction, shared_areas,
                               tag="summary", out_dir=FIG_OUT)
        plot_lag_difference_distribution(lag_df, direction,
                                          tag="summary", out_dir=FIG_OUT)

    # new Allen-ordered summary figures
    plot_lag_summary_pointplot(lag_df, shared_areas, out_dir=FIG_OUT)
    plot_lag_summary_heatmap(lag_df, shared_areas, out_dir=FIG_OUT)


# ---------------------------------------------------------------------------
# Main entry point — full pipeline
# ---------------------------------------------------------------------------

def run_analysis(unit_table: pd.DataFrame,
                 trial_table: pd.DataFrame,
                 learning_df: pd.DataFrame):
    """
    unit_table  : columns unit_id, mouse_id, session_id, probe,
                  area, bc_label, spike_times, reward_group
    trial_table : columns start_time, trial_type, context, session_id
    learning_df : columns mouse_id, learning_trial  (one row per mouse)
    """
    all_results = []
    for sid in unit_table["session_id"].unique():
        print(f"Processing session {sid} ...")
        u   = unit_table[unit_table["session_id"] == sid]
        t   = trial_table[trial_table["session_id"] == sid]
        res = analyse_session(u, t)
        df  = save_results(res, sid)
        all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)

    if "reward_group" not in results_df.columns:
        rg = unit_table[["unit_id", "reward_group"]].drop_duplicates()
        results_df = results_df.merge(rg, on="unit_id", how="left")

    lag_df = compute_lags(results_df, learning_df)
    if "reward_group" not in lag_df.columns:
        rg = unit_table[["mouse_id", "reward_group"]].drop_duplicates()
        lag_df = lag_df.merge(rg, on="mouse_id", how="left")

    # save CSVs first (before figures, so they exist if figures crash)
    results_df.drop(columns=["fr_series"], errors="ignore").to_csv(
        BASE_OUT / "all_sessions_sigmoid_results.csv", index=False
    )
    lag_df.to_csv(BASE_OUT / "lag_analysis.csv", index=False)

    # per-mouse figures
    for mouse_id in sorted(results_df["mouse_id"].unique()):
        print(f"  Mouse figures: {mouse_id}")
        make_mouse_figures(mouse_id, results_df, lag_df)

    # summary example neurons (need fr_series, only available in-memory)
    for stim, tag in [("whisker", "whisker_evoked"), ("auditory", "auditory_evoked")]:
        plot_example_neurons(
            results_df[(results_df["trial_type"] == stim) &
                       (results_df["epoch"]     == "evoked")],
            tag=tag, out_dir=FIG_OUT
        )

    _make_summary_figures(results_df, lag_df)

    print(f"Done. Results saved to {BASE_OUT}")
    return results_df, lag_df


# ---------------------------------------------------------------------------
# Figures-only mode — load existing CSVs, regenerate all figures
# ---------------------------------------------------------------------------

def run_figures_only(learning_df: pd.DataFrame):
    """
    Load existing per-session CSVs from BASE_OUT and regenerate all figures.
    Example neurons are skipped (fr_series not saved to disk).
    learning_df : columns mouse_id, learning_trial
    """
    print("Loading existing results ...")
    results_df = load_all_results()

    # recompute lag (needs learning_df)
    lag_df = compute_lags(results_df, learning_df)

    # attach reward_group to lag_df if present in results_df
    if "reward_group" in results_df.columns and "reward_group" not in lag_df.columns:
        rg = results_df[["mouse_id", "reward_group"]].drop_duplicates()
        lag_df = lag_df.merge(rg, on="mouse_id", how="left")

    # warn for mice in learning_df not in results
    missing = set(learning_df["mouse_id"]) - set(results_df["mouse_id"])
    for m in sorted(missing):
        print(f"WARNING: no results found for mouse {m}")

    # per-mouse figures (no example neurons — fr_series not on disk)
    for mouse_id in sorted(results_df["mouse_id"].unique()):
        print(f"  Mouse figures: {mouse_id}")
        #make_mouse_figures(mouse_id, results_df, lag_df)

    _make_summary_figures(results_df, lag_df)

    # save updated lag CSV
    lag_df.to_csv(BASE_OUT / "lag_analysis.csv", index=False)
    print(f"Done. Figures saved to {FIG_OUT}")
    return results_df, lag_df


# ---------------------------------------------------------------------------
# Learning curve loader
# ---------------------------------------------------------------------------

def load_learning_curves_data(path_to_data, subject_ids) -> pd.DataFrame:
    data = []
    for m in subject_ids:
        try:
            fname = f"{m}_whisker_0_whisker_trial_learning_curve_interp.h5"
            df_w  = pd.read_hdf(os.path.join(
                path_to_data, m, "whisker_0", "learning_curve", fname
            ))
            df_w["mouse_id"] = m
            data.append(df_w)
        except FileNotFoundError:
            print(f"No whisker curve for: {m}")
    return pd.concat(data).reset_index(drop=True)


def get_learning_df(path_to_data, subject_ids) -> pd.DataFrame:
    """Returns [mouse_id, learning_trial], one row per mouse."""
    data_df = load_learning_curves_data(path_to_data, subject_ids)
    corrections = {
        'MH036': 20, 'MH035': 20, 'MH031': 18, 'MH030': 34, 'MH028': 12,
        'MH011': 31, 'AB163': 15, 'AB159': 44, 'AB153': 20, 'AB141': np.nan,
        'AB140': 20, 'AB104': 25, 'AB102': 20, 'AB094': 20, 'AB093': 15,
        'AB116': 79,
    }
    for m_id, val in corrections.items():
        data_df.loc[data_df["mouse_id"] == m_id, "learning_trial"] = val
    return data_df.groupby("mouse_id")["learning_trial"].first().reset_index()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Full pipeline:
    #   learning_df = get_learning_df(path_to_data, subject_ids)
    #   run_analysis(unit_table, trial_table, learning_df)
    #
    # Figures only (reuse saved CSVs):
    #   learning_df = get_learning_df(path_to_data, subject_ids)
    #   run_figures_only(learning_df)
    print("Import and call run_analysis() or run_figures_only().")