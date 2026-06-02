"""
label_gui.py
------------
Minimal tkinter GUI to manually label GOOD/MUA neurons as OK or NOISE.

Usage
-----
    from label_gui import run_labeling_gui
    run_labeling_gui(unit_table, trial_table, output_csv="labels.csv")

Labels are appended to `output_csv` after every button press so you can
quit at any time without losing work.  Already-labeled units are skipped
on the next run.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from _shared import LABEL_COLS, plot_unit


def run_labeling_gui(
    unit_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    output_csv: str = "labels.csv",
    random_seed: int = 0,
) -> None:
    """
    Iterate over GOOD/MUA units in random order, show raster + waveform,
    and let the user label each as OK or NOISE.

    Parameters
    ----------
    unit_table  : must have columns in LABEL_COLS, 'spike_times',
                  'waveform_mean' (optional), 'bc_label'.
    trial_table : must have column 'start_time'.
    output_csv  : labels are appended here; existing labels are skipped.
    """
    # ── filter to GOOD/MUA; skip already-labeled units ────────────────────────
    candidates = unit_table[unit_table["bc_label"].isin(["GOOD", "MUA"])].copy()
    candidates = candidates.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    try:
        done = pd.read_csv(output_csv)[LABEL_COLS]
        candidates = candidates.merge(done.assign(_seen=True), on=LABEL_COLS,
                                      how="left")
        candidates = candidates[candidates["_seen"].isna()].drop(columns="_seen")
        candidates = candidates.reset_index(drop=True)
        print(f"  Skipping {len(done)} already-labeled units.")
    except FileNotFoundError:
        pass

    if candidates.empty:
        print("All units already labeled.")
        return

    trial_starts = np.sort(trial_table["start_time"].values)

    # ── state ─────────────────────────────────────────────────────────────────
    state = {"idx": 0}
    total = len(candidates)

    # ── GUI setup ─────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Noise labeler")

    fig, (ax_raster, ax_wave) = plt.subplots(
        1, 2, figsize=(11, 4),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.tight_layout(pad=2.5)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # status bar
    status_var = tk.StringVar()
    ttk.Label(root, textvariable=status_var, font=("Helvetica", 10)).pack(pady=2)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=6)

    # ── draw current unit ──────────────────────────────────────────────────────
    def draw(idx):
        ax_raster.cla()
        ax_wave.cla()
        row = candidates.iloc[idx]

        spikes   = np.asarray(row["spike_times"])
        waveform = row.get("waveform_mean", None)
        bc       = row["bc_label"]
        uid_info = "  ".join(str(row[c]) for c in LABEL_COLS)

        plot_unit(ax_raster, ax_wave, spikes, trial_starts, waveform)
        ax_raster.set_title(f"{uid_info}   bc_label={bc}", fontsize=8, loc="left")
        ax_wave.set_title("Mean waveform", fontsize=8)
        fig.suptitle(f"Unit {idx + 1} / {total}", fontsize=9, y=1.01)
        canvas.draw()
        status_var.set(f"{idx + 1} / {total}  |  {total - idx - 1} remaining")

    # ── save label and advance ─────────────────────────────────────────────────
    def label(value: str):
        row = candidates.iloc[state["idx"]]
        record = {c: row[c] for c in LABEL_COLS}
        record["manual_label"] = value          # "OK" or "NOISE"
        record["bc_label"]     = row["bc_label"]

        pd.DataFrame([record]).to_csv(
            output_csv, mode="a",
            header=not pd.io.common.file_exists(output_csv),
            index=False,
        )
        advance()

    def advance():
        state["idx"] += 1
        if state["idx"] >= total:
            status_var.set("Done!  All units labeled.")
            root.after(1500, root.destroy)
        else:
            draw(state["idx"])

    # ── buttons ───────────────────────────────────────────────────────────────
    ttk.Button(btn_frame, text="✓  OK",    width=14,
               command=lambda: label("OK")).grid(row=0, column=0, padx=8)
    ttk.Button(btn_frame, text="✗  NOISE", width=14,
               command=lambda: label("NOISE")).grid(row=0, column=1, padx=8)
    ttk.Button(btn_frame, text="→  Skip",  width=14,
               command=advance).grid(row=0, column=2, padx=8)

    # keyboard shortcuts
    root.bind("o", lambda _: label("OK"))
    root.bind("n", lambda _: label("NOISE"))
    root.bind("s", lambda _: advance())

    draw(0)
    root.mainloop()
