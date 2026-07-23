"""
label_gui.py — label neurons as OK or NOISE.

Controls: ◀/▶ toggle label · Enter confirm · Backspace go back · Close saves.
Output  : <output_dir>/labels.csv  (append-only, dedup on LABEL_COLS)
"""

import atexit, os, tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from noise_classification.hared import LABEL_COLS, plot_unit

AUTOSAVE_EVERY = 10


# ── Unit selection ─────────────────────────────────────────────────────────────

def _oddness(wf) -> float:
    """Waveform suspiciousness score in [0,1]."""
    if wf is None: return 0.5
    wf = np.asarray(wf, float)
    if len(wf) < 5 or np.all(wf == 0): return 0.5
    snr     = (wf.max() - wf.min()) / (wf[:max(1,len(wf)//10)].std() + 1e-9)
    ti      = int(np.argmin(wf)); half = wf[ti] / 2
    tw      = np.searchsorted(wf[ti:] > half, True) - \
              (ti - np.searchsorted(wf[:ti][::-1] > half, True))
    zc      = int(np.sum(np.diff(np.sign(wf)) != 0))
    return float((np.clip(1 - snr/20, 0, 1)
                + np.clip(1 - tw/10,  0, 1)
                + np.clip(zc/10,      0, 1)) / 3)


def _rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top-n most suspicious good/mua units by n_spikes, waveform, duration."""
    c = df[df["bc_label"].isin(["good", "mua"])].copy()
    if c.empty:
        raise ValueError(f"No good/mua units. bc_label values: {df['bc_label'].unique()}")

    ns  = (c["bc_nSpikes"].fillna(0) if "bc_nSpikes" in c.columns
           else c["spike_times"].apply(len)).values.astype(float)
    dur = c["spike_times"].apply(lambda s: s[-1]-s[0] if len(s)>1 else 0.).values.astype(float)

    score = np.zeros(len(c))
    for v, p in [(ns, 95), (dur, 95)]:
        p95 = np.percentile(v, p)
        if p95 > 0: score += np.clip(1 - v/p95, 0, 1)
    if "waveform_mean" in c.columns:
        score += c["waveform_mean"].apply(_oddness).values

    c["_s"] = score
    return c.sort_values("_s", ascending=False).head(n).drop(columns="_s").reset_index(drop=True)


# ── GUI ────────────────────────────────────────────────────────────────────────

def run_labeling_gui(
    unit_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    output_dir: str = "noise_classification/",
    n_units: int = 500,
    pre_ranked_candidates: pd.DataFrame | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "labels.csv")

    cands = pre_ranked_candidates.copy().reset_index(drop=True) \
            if pre_ranked_candidates is not None else _rank(unit_table, n_units)

    # skip already-labeled
    try:
        done = pd.read_csv(out_csv)[LABEL_COLS]
        for col in LABEL_COLS:
            if col in cands.columns:
                done[col] = done[col].astype(cands[col].dtype)
        cands = cands.merge(done.assign(_seen=True), on=LABEL_COLS, how="left")
        cands = cands[cands["_seen"].isna()].drop(columns="_seen").reset_index(drop=True)
    except FileNotFoundError:
        pass

    if cands.empty:
        print("All units already labeled."); return

    trial_starts = np.sort(trial_table["start_time"].values)
    total        = len(cands)
    pending: dict[int, str] = {}
    n_since_save = [0]

    def _flush():
        if not pending: return
        rows = [{**{c: cands.iloc[i][c] for c in LABEL_COLS},
                 "bc_label": cands.iloc[i]["bc_label"],
                 "manual_label": lbl}
                for i, lbl in pending.items()]
        new = pd.DataFrame(rows)
        try:
            new = pd.concat([pd.read_csv(out_csv), new]).drop_duplicates(LABEL_COLS, keep="last")
        except FileNotFoundError:
            pass
        new.to_csv(out_csv, index=False)
        n_since_save[0] = 0

    atexit.register(_flush)

    root = tk.Tk(); root.title("Noise labeler")
    fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(11, 4),
                                      gridspec_kw={"width_ratios": [3, 1]})
    fig.tight_layout(pad=2.5)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    ind_var = tk.StringVar()
    ind_lbl = tk.Label(root, textvariable=ind_var, font=("Helvetica", 16, "bold"), pady=4)
    ind_lbl.pack()
    sta_var = tk.StringVar()
    tk.Label(root, textvariable=sta_var, font=("Helvetica", 9), fg="grey").pack()
    tk.Label(root, text="◀/▶ toggle   Enter confirm   Backspace back",
             font=("Helvetica", 8), fg="#888").pack(pady=(0, 6))

    state = {"idx": 0, "label": "OK"}

    def _refresh_indicator():
        ok = state["label"] == "OK"
        ind_var.set("  ✓  OK  " if ok else "  ✗  NOISE  ")
        ind_lbl.config(bg="#1a7a1a" if ok else "#9b1c1c", fg="white")

    def _draw(idx):
        ax_r.cla(); ax_w.cla()
        row = cands.iloc[idx]
        plot_unit(ax_r, ax_w, np.asarray(row["spike_times"]),
                  trial_starts, row.get("waveform_mean"))
        ax_r.set_title("  ".join(str(row[c]) for c in LABEL_COLS)
                        + f"   bc_label={row['bc_label']}", fontsize=8, loc="left")
        fig.suptitle(f"Unit {idx+1}/{total}"
                     + (f"  [prev: {pending[idx]}]" if idx in pending else ""),
                     fontsize=9, y=1.01)
        canvas.draw()
        sta_var.set(f"{idx+1}/{total}  |  {sum(v=='NOISE' for v in pending.values())} NOISE  "
                    f"{sum(v=='OK' for v in pending.values())} OK")
        state["label"] = pending.get(idx, "OK")
        _refresh_indicator()

    def _toggle(e=None):
        state["label"] = "NOISE" if state["label"] == "OK" else "OK"
        _refresh_indicator()

    def _confirm(e=None):
        pending[state["idx"]] = state["label"]
        n_since_save[0] += 1
        if n_since_save[0] >= AUTOSAVE_EVERY: _flush()
        if state["idx"] + 1 >= total:
            _flush(); sta_var.set("Done!"); root.after(1500, root.destroy)
        else:
            state["idx"] += 1; _draw(state["idx"])

    def _back(e=None):
        if state["idx"] > 0: state["idx"] -= 1; _draw(state["idx"])

    root.bind("<Left>",      _toggle)
    root.bind("<Right>",     _toggle)
    root.bind("<Return>",    _confirm)
    root.bind("<BackSpace>", _back)
    root.protocol("WM_DELETE_WINDOW", lambda: (_flush(), root.destroy()))

    _draw(0); root.mainloop()
