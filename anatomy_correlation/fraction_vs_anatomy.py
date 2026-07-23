"""
fraction_vs_anatomy.py
──────────────────────
Correlate the fraction of significantly encoding units per brain area
against anatomical variables (cortical hierarchy score, axonal innervation
from S1), separately for ROC and GLM results.

Pipeline
--------
1. Load mouse metadata
2. Load unit table from NWB files -> area labels
3. Load ROC **or** GLM results; merge area labels from unit table
4. Filter / keep shared areas (R+ ∩ R-)
      - only bc_label == 'good' for unit/mouse counts
      - areas with total good units < N_MIN_GOOD_UNITS_TOTAL are removed
5. Compute fraction significant pooled across all units per area
      ROC : compute_prop_significant(..., per_subject=False)
            separately for 'positive' and 'negative' directions
      GLM : mean(significant) across all units per area x model
6. Remove areas with zero modulated units for a given variable
7. Merge anatomical variables onto frac_df via ccf_acronym_no_layer
8. Plot: for every encoding variable ->
      Figure A  : R+ and R- overlaid on the same axes
      Figure B  : separate subpanels per reward group
      Figure C  : pooled across reward groups
   ROC additionally produces figures split by direction (positive / negative).
   All panels are square with y-limits fitted to data range.
   Correlation annotations use Pearson r (rP) and Spearman r (rS).
"""

# ── standard library ──────────────────────────────────────────────────────────
import os
import socket
import pathlib

# ── scientific stack ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── custom utilities ──────────────────────────────────────────────────────────
import allen_utils as allen
import neural_utils as nutils
import plotting_utils
from roc_analysis.roc_analysis_utils import load_roc_results, compute_prop_significant

# ── column names produced by the two merge helpers ────────────────────────────
LIU_COL       = 'avg_ipsi_corr'      # added by merge_liu_avg_ipsi_opt
HIERARCHY_COL = 'cc_tc_ct_iterated'  # added by merge_hierarchy_from_harris

ANATOMICAL_VARS = {
    HIERARCHY_COL : 'Cortical hierarchy score\n(Harris et al.)',
    LIU_COL       : 'Axonal innervation from S1\n(Liu et al., ipsi avg, log)',
}

RG_COLORS  = {'R+': 'forestgreen', 'R-': 'crimson'}
DIR_COLORS = {'positive': 'tomato', 'negative': 'dodgerblue'}

CCF_PARENT_COL = 'ccf_atlas_parent_acronym'


# ── helper: area filtering ─────────────────────────────────────────────────────

def keep_shared_areas(data_df, nomenclature, n_min_units=5, n_min_mice=2,
                      n_min_good_total=30):
    """
    Retain only areas present in **both** reward groups that satisfy:
      - >= n_min_units good units per reward group
      - >= n_min_mice mice per reward group
      - >= n_min_good_total good units in total (across both reward groups)
    Only bc_label == 'good' is used for all counts.
    """
    print(f'Filtering shared areas [{nomenclature}]  '
          f'(>={n_min_units} good units, >={n_min_mice} mice per RG, '
          f'>={n_min_good_total} good units total)...')

    n_before  = len(data_df)
    data_df   = data_df.dropna(subset=[nomenclature])
    n_dropped = n_before - len(data_df)
    if n_dropped:
        print(f'  Dropped {n_dropped} rows with NaN in "{nomenclature}".')

    areas_rplus  = set(data_df[data_df['reward_group'] == 'R+'][nomenclature].unique())
    areas_rminus = set(data_df[data_df['reward_group'] == 'R-'][nomenclature].unique())
    candidates   = areas_rplus & areas_rminus

    # Use only good units for all thresholds
    good = data_df[data_df['bc_label'] == 'good']

    def _counts(rg):
        sub = good[good['reward_group'] == rg]
        u = sub.groupby(nomenclature)['unit_id'].nunique()
        m = sub.groupby(nomenclature)['mouse_id'].nunique()
        return u, m

    u_rp, m_rp = _counts('R+')
    u_rm, m_rm = _counts('R-')

    # Total good units across both reward groups
    u_total = good.groupby(nomenclature)['unit_id'].nunique()

    shared = [
        a for a in candidates
        if (u_rp.get(a, 0) >= n_min_units
            and u_rm.get(a, 0) >= n_min_units
            and m_rp.get(a, 0) >= n_min_mice
            and m_rm.get(a, 0) >= n_min_mice
            and u_total.get(a, 0) >= n_min_good_total)
    ]


    removed = candidates - set(shared)
    if removed:
        print(f'  Removed {len(removed)} areas with insufficient counts:')
        for a in sorted(removed):
            print(f'    {a}: R+ {u_rp.get(a,0)}u/{m_rp.get(a,0)}m  '
                  f'R- {u_rm.get(a,0)}u/{m_rm.get(a,0)}m  '
                  f'total_good={u_total.get(a,0)}')

    print(f'  Keeping {len(shared)} shared areas: {sorted(shared)}')
    for a in sorted(shared):
        print(f'    {a}: R+ {u_rp.get(a, 0)}u/{m_rp.get(a, 0)}m  '
              f'R- {u_rm.get(a, 0)}u/{m_rm.get(a, 0)}m  '
              f'total_good={u_total.get(a, 0)}')


    return data_df[data_df[nomenclature].isin(shared)], shared


# ── helper: pooled fraction (GLM) ─────────────────────────────────────────────

def compute_glm_fraction_pooled(data_df, area_col='ccf_acronym_no_layer'):
    """
    Fraction of significantly modulated units pooled across all units per
    reward_group x model_name x area. Excludes the 'full' model.
    Areas with proportion_all == 0 (no modulated units) are kept here
    and filtered later per variable in the plotting step.
    """
    sub = data_df[data_df['model_name'] != 'full'].copy()
    frac = (
        sub.groupby(['reward_group', 'model_name', area_col])
        .agg(
            proportion_all=('significant', 'mean'),
            n_units       =('significant', 'size'),
            n_mice        =('mouse_id',    'nunique'),
        )
        .reset_index()
        .rename(columns={'model_name': 'analysis_type'})
    )
    return frac


# ── helper: pooled fraction (ROC) ─────────────────────────────────────────────

def compute_roc_fraction_pooled(roc_df, area_col='ccf_acronym_no_layer'):
    """
    Wraps compute_prop_significant(per_subject=False).
    Returns all directions ('positive', 'negative') so the caller can
    plot them separately.
    Columns: reward_group, analysis_type, direction, area_col,
             proportion_all, proportion_signed, n_units.
    """
    frac = compute_prop_significant(roc_df, area_col=area_col, per_subject=False)
    return frac


# ── core plotting function ─────────────────────────────────────────────────────

def plot_fraction_vs_anatomical(frac_df, anat_col, x_label,
                                model_name_dict, saving_path,
                                area_col='ccf_acronym_no_layer',
                                fraction_col='proportion_all',
                                direction=None):
    """
    Produce three multi-panel figures (A: overlaid, B: separate panels,
    C: pooled) correlating fraction significant vs an anatomical variable.

    Parameters
    ----------
    frac_df         : DataFrame with reward_group, analysis_type, area_col,
                      fraction_col, n_units, anat_col already merged.
    anat_col        : str  x-axis anatomical column
    x_label         : str  x-axis label
    model_name_dict : dict  raw analysis_type -> display label
    saving_path     : str  output directory
    fraction_col    : str  y-axis column
    direction       : str or None
                      If provided (e.g. 'positive', 'negative'), appended to
                      figure names and titles (ROC use).
    """
    os.makedirs(saving_path, exist_ok=True)

    models = [m for m in model_name_dict if m in frac_df['analysis_type'].unique()]
    if not models:
        print(f'  [WARNING] No matching models for {anat_col}. Skipping.')
        return

    dir_suffix = f'_{direction}' if direction else ''
    dir_title  = f' ({direction})' if direction else ''

    def _fmt_p(p):
        return 'p<0.001' if p < 0.001 else f'p={p:.3f}'

    def _draw(ax, sub, color, label=None):
        """
        Scatter + regression on *ax*.
        Returns (rPearson, pPearson, rSpearman, pSpearman) or None.
        """
        sub = sub.dropna(subset=[anat_col, fraction_col])
        # Remove areas with zero modulated units
        sub = sub[sub[fraction_col] > 0]
        if len(sub) < 3:
            return None

        x     = sub[anat_col].values
        y     = sub[fraction_col].values
        sizes = 35 + 12 * np.log1p(sub['n_units'].values)

        ax.scatter(x, y, c=color, s=sizes, alpha=0.75,
                   edgecolors='white', linewidths=0.4, zorder=3, label=label)

        for _, row in sub.iterrows():
            ax.annotate(row[area_col],
                        xy=(row[anat_col], row[fraction_col]),
                        fontsize=5.5, alpha=0.55,
                        xytext=(3, 3), textcoords='offset points')

        slope, intercept, rP, pP, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color=color, lw=1.5, ls='--', alpha=0.85, zorder=2)

        rS, pS = stats.spearmanr(x, y)
        return rP, pP, rS, pS

    def _style(ax, sub_list, title, xlabel, ylabel):
        ax.set_title(title, fontsize=7.5, pad=4)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        plotting_utils.remove_top_right_frame(ax)

        # y-limits: 0 to data max + 10% padding (only non-zero rows)
        all_y = pd.concat([
            s.dropna(subset=[fraction_col])[fraction_col]
            for s in sub_list if not s.empty
        ], ignore_index=True)
        all_y = all_y[all_y > 0]
        if len(all_y):
            ax.set_ylim(0, all_y.max() * 1.12)

        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    n_models   = len(models)
    n_cols     = min(4, n_models)
    n_rows     = int(np.ceil(n_models / n_cols))
    panel_size = 3.5

    fig_A, axes_A = plt.subplots(n_rows, n_cols,
                                  figsize=(panel_size * n_cols,
                                           panel_size * n_rows), dpi=300)
    fig_B, axes_B = plt.subplots(n_rows, n_cols * 2,
                                  figsize=(panel_size * n_cols * 2,
                                           panel_size * n_rows), dpi=300)
    axes_A = np.array(axes_A).flatten()
    axes_B = np.array(axes_B).flatten()

    for idx, model in enumerate(models):
        display = model_name_dict.get(model, model)

        # Figure A: overlaid
        ax_a        = axes_A[idx]
        stats_lines = []
        subs_a      = []
        for rg in ['R+', 'R-']:
            sub = frac_df[(frac_df['analysis_type'] == model) &
                          (frac_df['reward_group']  == rg)]
            subs_a.append(sub)
            res = _draw(ax_a, sub, RG_COLORS[rg], label=rg)
            if res:
                rP, pP, rS, pS = res
                stats_lines.append(
                    f"{rg}  Pearson r={rP:.2f}({_fmt_p(pP)})  "
                    f"Spearman r={rS:.2f}({_fmt_p(pS)})"
                )
        ax_a.legend(fontsize=6.5, frameon=False)
        _style(ax_a, subs_a,
               title=f"{display}{dir_title}\n" + "\n".join(stats_lines),
               xlabel=x_label, ylabel='Fraction significant')

        # Figure B: separate panels per reward group
        for rg_idx, rg in enumerate(['R+', 'R-']):
            ax_b = axes_B[idx * 2 + rg_idx]
            sub  = frac_df[(frac_df['analysis_type'] == model) &
                           (frac_df['reward_group']  == rg)]
            res  = _draw(ax_b, sub, RG_COLORS[rg])
            corr_str = ''
            if res:
                rP, pP, rS, pS = res
                corr_str = (f"Pearson r={rP:.2f} ({_fmt_p(pP)})\n"
                            f"Spearman r={rS:.2f} ({_fmt_p(pS)})")
            _style(ax_b, [sub],
                   title=f"{display}{dir_title} -- {rg}\n{corr_str}",
                   xlabel=x_label, ylabel='Fraction significant')

    for i in range(n_models, len(axes_A)):
        axes_A[i].set_visible(False)
    for i in range(n_models * 2, len(axes_B)):
        axes_B[i].set_visible(False)

    for fig, tag in [(fig_A, 'overlaid'), (fig_B, 'separate_panels')]:
        fig.suptitle(
            f'Fraction significant{dir_title} vs '
            f'{x_label.split(chr(10))[0]}  [{tag}]',
            fontsize=10, y=1.01)
        fig.tight_layout()
        plotting_utils.save_figure_with_options(
            fig, ['png', 'pdf', 'svg'],
            f'frac_sig_vs_{anat_col}_{tag}{dir_suffix}',
            saving_path, dark_background=False)
        plt.close(fig)

    # Figure C: pooled R+/R-
    pooled = (
        frac_df.groupby(['analysis_type', area_col])
        .agg(proportion_all=('proportion_all', 'mean'),
             n_units       =('n_units',        'sum'))
        .reset_index()
    )
    anat_map = (frac_df[[area_col, anat_col]]
                .dropna(subset=[anat_col])
                .drop_duplicates(area_col)
                .set_index(area_col)[anat_col])
    pooled[anat_col] = pooled[area_col].map(anat_map)

    fig_C, axes_C = plt.subplots(n_rows, n_cols,
                                  figsize=(panel_size * n_cols,
                                           panel_size * n_rows), dpi=300)
    axes_C = np.array(axes_C).flatten()

    for idx, model in enumerate(models):
        display  = model_name_dict.get(model, model)
        ax_c     = axes_C[idx]
        sub      = pooled[pooled['analysis_type'] == model].dropna(subset=[anat_col])
        res      = _draw(ax_c, sub, 'steelblue')
        corr_str = ''
        if res:
            rP, pP, rS, pS = res
            corr_str = (f"Pearson r={rP:.2f} ({_fmt_p(pP)})\n"
                        f"Spearman r={rS:.2f} ({_fmt_p(pS)})")
        _style(ax_c, [sub],
               title=f"{display}{dir_title} (pooled R+/R-)\n{corr_str}",
               xlabel=x_label, ylabel='Fraction significant')

    for i in range(n_models, len(axes_C)):
        axes_C[i].set_visible(False)

    fig_C.suptitle(
        f'Fraction significant{dir_title} (pooled R+/R-) vs '
        f'{x_label.split(chr(10))[0]}',
        fontsize=10, y=1.01)
    fig_C.tight_layout()
    plotting_utils.save_figure_with_options(
        fig_C, ['png', 'pdf', 'svg'],
        f'frac_sig_vs_{anat_col}_pooled{dir_suffix}',
        saving_path, dark_background=False)
    plt.close(fig_C)

    frac_df.to_csv(
        os.path.join(saving_path, f'frac_sig_vs_{anat_col}{dir_suffix}.csv'),
        index=False)
    print(f'  Saved -> {saving_path}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ====================================================================
    # USER SETTINGS
    # ====================================================================
    MODE             = 'glm'      # 'roc'  or  'glm'

    GLM_EXPERIMENTER = 'Myriam_Hamon'
    GIT_VERSION      = '1b14083'  # 'f849441' or '1b14083'

    N_UNITS_MIN           = 5    # min good units per area per reward group
    N_MICE_PER_AREA_MIN   = 2    # min mice per area per reward group
    N_MIN_GOOD_UNITS_TOTAL = 30  # min good units total (both reward groups)
    # ====================================================================

    # ── 1. Paths ──────────────────────────────────────────────────────────────
    hostname = socket.gethostname()
    if 'haas' in hostname:
        DATA_PATH       = pathlib.Path('/mnt/lsens-analysis/')
        AXEL_NWB_PATH   = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/NWB_combined')
        MYRIAM_NWB_PATH = pathlib.Path('/mnt/lsens-analysis/Myriam_Hamon/NWB')
        FIGURE_PATH     = pathlib.Path('/mnt/lsens-analysis/Axel_Bisi/combined_results/fraction_vs_anatomy')
        INFO_PATH       = pathlib.Path('/mnt/share_internal/Axel_Bisi_Share/dataset_info')
    else:
        DATA_PATH       = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis')
        AXEL_NWB_PATH   = pathlib.Path(r'M:\analysis\Axel_Bisi\NWB_combined')
        MYRIAM_NWB_PATH = pathlib.Path(r'M:\analysis\Myriam_Hamon\NWB')
        FIGURE_PATH     = pathlib.Path(r'M:\analysis\Axel_Bisi\combined_results\fraction_vs_anatomy')
        INFO_PATH       = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info')

    os.makedirs(FIGURE_PATH, exist_ok=True)

    # ── 2. Mouse metadata ─────────────────────────────────────────────────────
    print('Loading mouse metadata...')
    mouse_info_df = pd.read_excel(INFO_PATH / 'joint_mouse_reference_weight.xlsx')
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude']       == 0) &
        (mouse_info_df['exclude_ephys'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-'])) &
        (mouse_info_df['recording']     == 1)
    ]
    valid_mice = mouse_info_df['mouse_id'].unique()

    # ── 3. Unit table ─────────────────────────────────────────────────────────
    print('Loading NWB unit table...')
    nwb_list = []
    for m in valid_mice:
        path = AXEL_NWB_PATH if m.startswith('AB') else (
               MYRIAM_NWB_PATH if m.startswith('MH') else None)
        if path is None:
            continue
        nwb_list.extend(str(path / f) for f in os.listdir(path) if m in f)

    _, unit_table, _ = nutils.combine_ephys_nwb(nwb_list, max_workers=8)
    unit_table = unit_table[unit_table['bc_label'] == 'good']
    unit_table = allen.process_allen_labels(unit_table, subdivide_areas=True)
    print(f'  {len(unit_table)} good units | {unit_table["mouse_id"].nunique()} mice')

    UNIT_COLS = ['mouse_id', 'neuron_id', 'unit_id', 'bc_label',
                 'area_acronym_custom', 'ccf_acronym_no_layer', CCF_PARENT_COL]
    AREA_COL  = 'ccf_acronym_no_layer'

    # ── 4. Load metric data, merge area labels, filter areas ─────────────────

    if MODE == 'roc':
        print('Loading ROC results...')
        roc_data_path = DATA_PATH / 'Axel_Bisi' / 'combined_results'
        roc_df = load_roc_results(roc_data_path, max_workers=8)
        roc_df = roc_df[roc_df['mouse_id'].isin(unit_table['mouse_id'].unique())]
        roc_df = roc_df.merge(
            mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')
        roc_df = roc_df.drop(
            columns=list(set(UNIT_COLS) - {'mouse_id', 'neuron_id'}), errors='ignore')
        roc_df = roc_df.merge(unit_table[UNIT_COLS], on=['mouse_id', 'neuron_id'], how='left')

        roc_df, _ = keep_shared_areas(
            roc_df, 'ccf_acronym_no_layer',
            n_min_units=N_UNITS_MIN,
            n_min_mice=N_MICE_PER_AREA_MIN,
            n_min_good_total=N_MIN_GOOD_UNITS_TOTAL)

        print('Computing pooled fractions (ROC)...')
        # Returns all directions; we'll loop over them when plotting
        frac_df = compute_roc_fraction_pooled(roc_df, area_col=AREA_COL)

        result_label = 'roc'
        MODEL_NAME_DICT = {
            'whisker_active'               : 'Whisker (active)',
            'auditory_active'              : 'Auditory (active)',
            'wh_vs_aud_active'             : 'Modality selective (active)',
            'choice'                       : 'Choice',
            'whisker_passive_pre'          : 'Whisker passive pre',
            'whisker_passive_post'         : 'Whisker passive post',
            'auditory_passive_pre'         : 'Auditory passive pre',
            'auditory_passive_post'        : 'Auditory passive post',
            'whisker_pre_vs_post_learning' : 'Whisker pre->post',
            'auditory_pre_vs_post_learning': 'Auditory pre->post',
        }
        DIRECTIONS = frac_df['direction'].unique().tolist()

    elif MODE == 'glm':
        print(f'Loading GLM results (git={GIT_VERSION})...')
        glm_data_path = DATA_PATH / GLM_EXPERIMENTER / 'combined_results'

        glm_dfs = []
        for mouse_id in valid_mice:
            fpath = (glm_data_path / mouse_id / 'whisker_0' / 'unit_glm'
                     / GIT_VERSION
                     / f'summary_{mouse_id}_unit_glm_{GIT_VERSION}.parquet')
            if not fpath.exists():
                print(f'  [WARNING] not found: {fpath}')
                continue
            glm_dfs.append(pd.read_parquet(fpath))
        if not glm_dfs:
            raise FileNotFoundError('No GLM parquet files found.')

        glm_df = pd.concat(glm_dfs, ignore_index=True)
        glm_df['significant']  = (glm_df['test_corr'] > 0.2) & glm_df['lrt_significant']
        glm_df['reward_group'] = glm_df['reward_group'].map({1: 'R+', 0: 'R-'})
        glm_df = glm_df.drop(
            columns=list(set(UNIT_COLS) - {'mouse_id', 'neuron_id'}), errors='ignore')
        glm_df = glm_df.merge(unit_table[UNIT_COLS], on=['mouse_id', 'neuron_id'], how='left')

        glm_df, _ = keep_shared_areas(
            glm_df, 'ccf_acronym_no_layer',
            n_min_units=N_UNITS_MIN,
            n_min_mice=N_MICE_PER_AREA_MIN,
            n_min_good_total=N_MIN_GOOD_UNITS_TOTAL)

        print('Computing pooled fractions (GLM)...')
        frac_df = compute_glm_fraction_pooled(glm_df, area_col=AREA_COL)

        result_label = f'glm_{GIT_VERSION}'
        DIRECTIONS   = [None]   # GLM has no direction split

        if GIT_VERSION == 'f849441':
            MODEL_NAME_DICT = {
                'auditory_encoding'    : 'Auditory stimulus',
                'whisker_encoding'     : 'Whisker stimulus',
                'jaw_onset_encoding'   : 'Lick initiation',
                'motor_encoding'       : 'Orofacial motion',
                'last_whisker_reward'  : 'Prev. whisker rewarded',
                'prev_success'         : 'Previous trial success',
                'block_perf_type'      : 'High/low performance',
                'sum_rewards'          : 'Cumulative rewards',
                'whisker_reward_rate_5': 'Perf. last 5 whisker trials',
            }
        else:  # 1b14083
            MODEL_NAME_DICT = {
                'auditory_encoding'        : 'Auditory stimulus',
                'whisker_encoding'         : 'Whisker stimulus',
                'jaw_onset_encoding'       : 'Lick initiation',
                'reward_encoding'          : 'Reward time',
                'motor_encoding'           : 'Orofacial motion',
                'pupil_area'               : 'Pupil area',
                'time_since_whisker_reward': 'Whisker reward recency',
                'block_perf_type'          : 'High/low performance',
                'session_progress_encoding': 'Trial index',
            }
    else:
        raise ValueError(f'Unknown MODE "{MODE}". Choose "roc" or "glm".')

    # ── 5. Merge anatomical variables ─────────────────────────────────────────
    print('Merging anatomical variables...')
    data_areas = frac_df[AREA_COL].dropna().unique()

    liu_areas = allen.load_liu_et_al_avg_ipsi()
    print(f'  Liu: {len(liu_areas)} areas | data: {len(data_areas)} | '
          f'intersect: {len(set(liu_areas.keys()) & set(data_areas))}')
    frac_df = allen.merge_liu_avg_ipsi_opt(frac_df, cols_priority=AREA_COL)
    frac_df[LIU_COL] = np.log(frac_df[LIU_COL] + 1e-5)

    harris_df    = allen.load_process_hierarchy_from_harris()
    harris_areas = harris_df[AREA_COL].dropna().unique()
    print(f'  Harris: {len(harris_areas)} areas | data: {len(data_areas)} | '
          f'intersect: {len(set(harris_areas) & set(data_areas))}')
    frac_df = allen.merge_hierarchy_from_harris(frac_df, merge_on=AREA_COL)

    for col, label in [(LIU_COL, 'Liu'), (HIERARCHY_COL, 'Harris')]:
        print(f'  {label} ({col}): {frac_df[col].notna().sum()}/{len(frac_df)} non-null')

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    for anat_col, x_label in ANATOMICAL_VARS.items():
        if frac_df[anat_col].isna().all():
            print(f'[WARNING] "{anat_col}" all-NaN -- skipping.')
            continue

        out_path = FIGURE_PATH / result_label / anat_col
        print(f'\nPlotting vs {anat_col}  ->  {out_path}')

        for direction in DIRECTIONS:
            # Subset by direction for ROC; no-op for GLM (direction=None)
            if direction is not None:
                plot_df = frac_df[frac_df['direction'] == direction].copy()
                # Use proportion_signed for directional ROC plots
                fcol = 'proportion_signed'
            else:
                plot_df = frac_df.copy()
                fcol    = 'proportion_all'

            plot_fraction_vs_anatomical(
                frac_df         = plot_df,
                anat_col        = anat_col,
                x_label         = x_label,
                model_name_dict = MODEL_NAME_DICT,
                saving_path     = str(out_path),
                area_col        = AREA_COL,
                fraction_col    = fcol,
                direction       = direction,
            )

    print('\nDone.')