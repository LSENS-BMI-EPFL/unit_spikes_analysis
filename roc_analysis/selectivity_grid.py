#!/usr/bin/env python3
"""
Selectivity proportion grid.

Accepts a long-format DataFrame with one row per
(brain_area × analysis_type × direction).

Quick start
-----------
    plot_selectivity(df, style='donut')   # ring charts
    plot_selectivity(df, style='bar')     # grouped vertical bars

ROW_CONFIG format  — 8-tuple per row:
  (row_label, pos_label, neg_label, analysis_type, pos_dir, neg_dir, pos_color, neg_color)

  analysis_type  — exact string in your df's analysis_type column for this row
  pos_dir        — exact string in your df's direction column for "positive" direction
  neg_dir        — exact string in your df's direction column for "negative" direction

Column name defaults (override via col_* kwargs in plot_selectivity):
  area_col      = 'ccf_acronym_no_layer'
  analysis_col  = 'analysis_type'
  direction_col = 'direction'
  value_col     = 'proportion'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import allen_utils
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit this section for your data)
# ─────────────────────────────────────────────────────────────────────────────

BRAIN_AREAS = allen_utils.get_custom_area_groups().keys()

# (row_label, pos_label, neg_label, analysis_type, pos_dir, neg_dir, pos_color, neg_color)
#
#  analysis_type  — must exactly match a value in your df's analysis_type column
#  pos_dir/neg_dir — must exactly match values in your df's direction column
ROW_CONFIG = [
    ('Whisker\nresponsive',  'Positive', 'Negative', 'whisker_active', 'positive', 'negative', '#E74C3C', '#3498DB'),
    ('Auditory\nresponsive', 'Positive', 'Negative', 'auditory_active','positive', 'negative', '#E74C3C', '#3498DB'),
    ('Modality\nselective',  'Whisker',  'Auditory', 'wh_vs_aud_active', 'whisker',  'auditory', '#ebb134', '#3127c2'),
    ('Choice\nselective',    'Positive', 'Negative', 'choice',         'positive', 'negative', '#E74C3C', '#3498DB'),
    ('Spontaneous\nlicks',   'Positive', 'Negative', 'spontaneous_licks','positive','negative','#E74C3C', '#3498DB'),
]

NONSIG_COLOR = '#CCCCCC'

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE DATA
# ─────────────────────────────────────────────────────────────────────────────

def make_example_df(row_config=ROW_CONFIG, brain_areas=BRAIN_AREAS, seed=42) -> pd.DataFrame:
    """Fake long-format DataFrame — replace with your real df."""
    rng = np.random.default_rng(seed)
    records = []
    for _, _, _, analysis, pos_dir, neg_dir, _, _ in row_config:
        for area in brain_areas:
            p = rng.beta(2, 8) * 0.55
            n = rng.beta(2, 8) * 0.55
            if p + n > 0.82:
                s = 0.80 / (p + n); p *= s; n *= s
            for direction, val in [(pos_dir, p), (neg_dir, n)]:
                records.append(dict(
                    analysis_type=analysis,
                    ccf_acronym_no_layer=area,
                    direction=direction,
                    proportion=val,
                ))
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _lookup(df, analysis, area, pos_dir, neg_dir,
            analysis_col, area_col, direction_col, value_col,
            nonsig_dir='non-selective', scale=100.0):
    """Return (pos, neg, nonsig) as fractions 0-1 for one (analysis, area) cell.

    scale : divide raw values by this to convert to 0-1
            (100 if your df stores percentages, 1 if already fractions)
    nonsig_dir : the direction label used for non-selective neurons in the df
    """
    sub = df[(df[analysis_col] == analysis) & (df[area_col] == area)]
    def get(d):
        row = sub[sub[direction_col] == d]
        return float(row[value_col].values[0]) / scale if len(row) else 0.0
    pos    = get(pos_dir)
    neg    = get(neg_dir)
    nonsig = get(nonsig_dir)
    # Fall back to computing nonsig if not present in df
    if nonsig == 0.0 and (pos + neg) < 1.0:
        nonsig = max(0.0, 1.0 - pos - neg)
    return pos, neg, nonsig

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _row_label_ax(fig, gs_cell, row_label, pos_lbl, neg_lbl, pc, nc):
    ax = fig.add_subplot(gs_cell)
    ax.axis('off')
    ax.text(0.97, 0.63, row_label,
            ha='right', va='center', transform=ax.transAxes,
            fontsize=7, color='#1a1a1a', multialignment='right', linespacing=1.3)
    ax.text(0.97, 0.27, pos_lbl,
            ha='right', va='center', transform=ax.transAxes,
            fontsize=6.5, color=pc, fontweight='bold')
    ax.text(0.97, 0.11, neg_lbl,
            ha='right', va='center', transform=ax.transAxes,
            fontsize=6.5, color=nc, fontweight='bold')
    return ax


def _nonsig(pos, neg):
    # Kept for compatibility — prefer looking up non-selective directly from df.
    return max(0.0, 1.0 - pos - neg)


def _finish_fig(fig, row_config, nonsig_color, title, savepath, bottom_pad):
    seen, handles = set(), []
    for _, pos_lbl, neg_lbl, _, _, _, pc, nc in row_config:
        for lbl, col in [(pos_lbl, pc), (neg_lbl, nc)]:
            if col not in seen:
                handles.append(mpatches.Patch(color=col, label=lbl))
                seen.add(col)
    handles.append(mpatches.Patch(color=nonsig_color, label='Non-significant'))
    fig.legend(handles=handles, loc='lower center',
               ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.56, bottom_pad))
    if title:
        fig.suptitle(title, y=0.99, fontsize=9, color='#1a1a1a')
    plt.show()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f'Saved → {savepath}')

# ─────────────────────────────────────────────────────────────────────────────
# CELL DRAWERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_donut(ax, pos, neg, nonsig, pc, nc, nonsig_color, min_wedge=5e-3):
    sizes  = [pos, neg, nonsig]
    colors = [pc,  nc,  nonsig_color]
    pairs  = [(s, c) for s, c in zip(sizes, colors) if s > min_wedge]
    if pairs:
        s_arr, c_arr = zip(*pairs)
        ax.pie(list(s_arr), colors=list(c_arr), startangle=90,
               wedgeprops={'width': 0.46, 'edgecolor': 'white', 'linewidth': 0.35},
               counterclock=False)
    ax.set_aspect('equal')


def _draw_bar(ax, pos, neg, nonsig, pc, nc, nonsig_color):
    heights = [pos, neg, nonsig]
    colors  = [pc,  nc,  nonsig_color]
    ax.bar([0, 1, 2], heights, width=0.7, color=colors,
           edgecolor='white', linewidth=0.4, zorder=2)
    ax.set_xlim(-0.65, 2.65)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(0.4)
    ax.spines['left'].set_color('#aaa')
    ax.tick_params(bottom=False, labelbottom=False,
                   left=True, labelleft=False,
                   length=2, width=0.4, color='#aaa')
    ax.set_yticks([0, 0.5, 1])
    ax.yaxis.grid(True, linewidth=0.3, color='#ddd', zorder=0)
    ax.set_axisbelow(True)



def _annotate_cell(ax, pos, neg, nonsig, pc, nc):
    """Draw pos / neg / nonsig percentages as a right-side text column."""
    # Use the axes bbox in figure coords; place text just outside right edge.
    # We attach to the axes using a blended transform: x in axes coords, y in data/axes.
    labels  = [f'{pos*100:.1f}%',   f'{neg*100:.1f}%',   f'{nonsig*100:.1f}%']
    colors  = [pc,               nc,               '#888']
    weights = ['bold',           'bold',           'normal']
    ys      = [0.78,             0.50,             0.22]   # axes-fraction heights
    for lbl, col, w, y in zip(labels, colors, weights, ys):
        ax.text(0.92, y, lbl,
                transform=ax.transAxes,
                ha='left', va='center',
                fontsize=3.5, color=col, fontweight=w)

# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def plot_selectivity(
        df,
        row_config    = ROW_CONFIG,
        brain_areas   = BRAIN_AREAS,
        nonsig_color  = NONSIG_COLOR,
        style         = 'donut',          # 'donut' | 'bar'
        title         = 'Selectivity across brain areas',
        savepath      = None,
        analysis_col  = 'analysis_type',
        area_col      = 'ccf_acronym_no_layer',
        direction_col = 'direction',
        value_col     = 'proportion',
        annotate      = True,          # show proportion values on each cell
) -> plt.Figure:
    """
    Plot a selectivity proportion grid from a long-format DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format, one row per (area × analysis_type × direction).
    row_config : list of 8-tuples
        (row_label, pos_label, neg_label, analysis_type,
         pos_dir, neg_dir, pos_color, neg_color)
    style : 'donut' | 'bar'
    """
    if style not in ('donut', 'bar'):
        raise ValueError(f"style must be 'donut' or 'bar', got {style!r}")

    n_rows = len(row_config)
    n_cols = len(brain_areas)
    LWIDTH = 1.55

    if style == 'donut':
        cell_h, wsp, hsp, top, bot, legend_pad = 1.0, 0.04, 0.08, 0.91, 0.07, -0.01
    else:
        cell_h, wsp, hsp, top, bot, legend_pad = 0.90, 0.08, 0.55, 0.87, 0.09, -0.04

    fig = plt.figure(figsize=(n_cols * 0.90 + LWIDTH * 0.90, n_rows * cell_h + 0.65))
    gs  = GridSpec(n_rows, n_cols + 1,
                   width_ratios=[LWIDTH] + [1] * n_cols,
                   wspace=wsp, hspace=hsp,
                   left=0.01, right=0.995, top=top, bottom=bot)

    for i, (row_label, pos_lbl, neg_lbl, analysis, pos_dir, neg_dir, pc, nc) in enumerate(row_config):
        _row_label_ax(fig, gs[i, 0], row_label, pos_lbl, neg_lbl, pc, nc)

        # Order of brain areas
        areas_ordered = ['Motor and frontal areas',
                       'Somatosensory areas',
                       'Auditory areas',
                       'Visual areas',
                       'Retrosplenial areas',
                       'Striatum and pallidum',
                       'Thalamus',
                       'Hippocampus',
                       #'Cortical subplate',
                       'Midbrain',
                       'Olfactory areas',
                       'Amygdala and hypothalamus'

        ]
        brain_areas_ordered = [area for area in brain_areas if area in areas_ordered]

        for j, area in enumerate(brain_areas_ordered):
            ax = fig.add_subplot(gs[i, j + 1])
            pos, neg, nonsig = _lookup(df, analysis, area,
                                          pos_dir, neg_dir,
                                          analysis_col, area_col, direction_col, value_col)

            if style == 'donut':
                ax.axis('off')
                _draw_donut(ax, pos, neg, nonsig, pc, nc, nonsig_color)
            else:
                _draw_bar(ax, pos, neg, nonsig, pc, nc, nonsig_color)
            if annotate:
                _annotate_cell(ax, pos, neg, nonsig, pc, nc)

            if i == 0:
                area = area if len(area) <= 12 else area.replace(' ', '\n', 1)
                label = f'{area}'
                ax.set_title(label, fontsize=5.5, pad=1.5, color='#444')

    _finish_fig(fig, row_config, nonsig_color, title, savepath, legend_pad)
    return fig


# Backwards-compatible aliases
def plot_selectivity_donuts(df, **kw): return plot_selectivity(df, style='donut', **kw)
def plot_selectivity_bars(df, **kw):   return plot_selectivity(df, style='bar',   **kw)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = make_example_df()
    plot_selectivity(df, style='donut', savepath='selectivity_donuts.pdf')
    plot_selectivity(df, style='bar',   savepath='selectivity_bars.pdf')
