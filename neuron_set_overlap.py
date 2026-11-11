"""Overlap of the neurons that different methods identify as significant, and
how the methods' effect sizes relate. Generic: any number of methods, each
given as one long table.

Results table (one row per neuron and test), the only input format:
    mouse_id, neuron_id   neuron identity (id_cols)
    method                e.g. 'GLM', 'ROC'
    test                  e.g. a reduced model ('whisker_encoding') or an ROC
                          analysis type ('whisker_active')
    significant           bool
    <effect columns>      any numeric columns, e.g. delta_test_corr,
                          abs_selectivity (NaN where they don't apply)
Build it with make_results() from any per-neuron table, or with the adapters
glm_results() (GLM summaries) and roc_results() (ROC csv files). Several
methods: pd.concat their tables.

Neuron table (info): id_cols + optional scope column (default reward_group,
for all / R+ / R- versions) and area column (default area_group). Only
neurons in info are used; each comparison uses the neurons with a result for
both of its tests.

Outputs of run_overlap(results, info, output_folder, ...):
  matrix_<A>_vs_<B>/   every test of method A x every test of method B: % of
                       A-significant also B-significant and the reverse, and
                       the Fisher log2 odds ratio (BH-FDR stars), per scope
  pairs/<A>-<test>__vs__<B>-<test>/
                       counts (both / A only / B only), overlap per area
                       group, and every effect of A vs every effect of B
                       (Spearman, pooled and median per mouse), each effect
                       split by the other test's significance
  pairs_overlap_stats.txt, pairs_effect_size_stats.txt
  upset/               intersections of several (method, test) sets

Presets: run_glm_vs_roc (GLM models vs ROC analysis types), run_glm_within
(reduced models of one fit against each other), run_glm_vs_glm (two fits,
e.g. two git versions). To run: edit the `runs` list at the bottom of this
file, then `python neuron_set_overlap.py`.
"""
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

import visualize_quantity as vq

ID_COLS = ['mouse_id', 'neuron_id']
RESERVED_COLS = {'method', 'test', 'significant'}
REWARD_SCOPES = [('all', None), ('R+', 1), ('R-', 0)]
MIN_NEURONS_AREA = 20
BOTH_COLOR, A_COLOR, B_COLOR, NEITHER_COLOR = '#6a3d9a', '#1f78b4', '#e66101', '#bdbdbd'


# ── building the results table ───────────────────────────────────────────────

def make_results(df, method, test_col, significant_col, effect_cols=(), id_cols=ID_COLS, rename_effects=None):
    """Results table of one method from any table with one row per neuron and
    test: test_col names the test, significant_col is the significance flag,
    effect_cols are kept as effect sizes (optionally renamed)."""
    out = df[list(id_cols) + [test_col, significant_col] + list(effect_cols)].copy()
    out = out.rename(columns={test_col: 'test', significant_col: 'significant', **(rename_effects or {})})
    out['significant'] = out['significant'].map(
        lambda v: str(v).lower() == 'true' if isinstance(v, str) else bool(v) if pd.notna(v) else False)
    out.insert(len(id_cols), 'method', method)
    return out.drop_duplicates(list(id_cols) + ['test']).reset_index(drop=True)


def glm_results(merged_df, method='GLM', id_cols=ID_COLS):
    """GLM summaries -> results: test = reduced model, significant =
    lrt_significant, effects = drop in test_corr and test_score from the full
    model (full - reduced)."""
    cols = list(id_cols) + ['model_name', 'test_corr', 'test_score', 'lrt_significant']
    df = merged_df[[c for c in cols if c in merged_df.columns]].drop_duplicates(list(id_cols) + ['model_name']).copy()
    full = df[df['model_name'] == 'full'].set_index(list(id_cols))
    red = df[df['model_name'] != 'full'].copy()
    idx = pd.MultiIndex.from_frame(red[list(id_cols)])
    effects = []
    for metric in ('test_corr', 'test_score'):
        if metric in red.columns:
            red[f'delta_{metric}'] = (full[metric].reindex(idx).to_numpy(dtype=float)
                                      - red[metric].to_numpy(dtype=float))
            effects.append(f'delta_{metric}')
    return make_results(red, method, 'model_name', 'lrt_significant', effects, id_cols)


def roc_results(mice, root=vq.ROC_ROOT, session_folder='whisker_0', method='ROC', max_workers=16):
    """ROC csv files ({root}/{mouse}/{session_folder}/roc_analysis/
    *_roc_results_new.csv) -> results: test = analysis_type, effects =
    abs_selectivity (|2 AUC - 1|) and selectivity."""
    read_cols = ID_COLS + ['analysis_type', 'significant', 'selectivity']

    def _read(mouse):
        # No glob here: the '?' of the '\\?\UNC\' long-path prefix is a glob wildcard
        # on older Pythons, which then silently match nothing.
        folder = vq._long_path(os.path.join(root, mouse, session_folder, 'roc_analysis'))
        try:
            names = [n for n in os.listdir(folder) if n.endswith('_roc_results_new.csv')]
        except OSError:
            names = []
        tables = []
        for name in names:
            path = os.path.join(folder, name)
            try:
                tables.append(pd.read_csv(path, usecols=lambda c: c in read_cols))
            except Exception as e:
                print(f"[overlap] Skipped {path} ({type(e).__name__}: {e})")
        return pd.concat(tables, ignore_index=True) if tables else None

    mice = sorted(set(mice))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        read = dict(zip(mice, pool.map(_read, mice)))
    missing = [m for m, t in read.items() if t is None]
    if missing:
        print(f"[overlap] No ROC results in {root}\\<mouse>\\{session_folder}\\roc_analysis for "
              f"{len(missing)}/{len(mice)} mice: {missing}")
    tables = [t for t in read.values() if t is not None]
    if not tables:
        return pd.DataFrame(columns=ID_COLS + ['method', 'test', 'significant'])
    roc = pd.concat(tables, ignore_index=True)
    roc['abs_selectivity'] = roc['selectivity'].abs()
    return make_results(roc, method, 'analysis_type', 'significant', ['abs_selectivity', 'selectivity'])


def neuron_info_from_glm(merged_df, area_group_from_name=None, excluded_area_groups=(), id_cols=ID_COLS):
    """Neuron table from GLM summaries: reward_group, area_acronym_custom and
    area_group (rows of excluded area groups dropped)."""
    info = merged_df[list(id_cols) + ['reward_group', 'area_acronym_custom']].drop_duplicates(list(id_cols)).copy()
    info['area_group'] = info['area_acronym_custom'].map(area_group_from_name or {})
    excluded = info['area_group'].fillna('').str.lower().isin({g.lower() for g in excluded_area_groups})
    return info[~excluded].reset_index(drop=True)


def _normalise_ids(df, id_cols):
    df = df.copy()
    for c in id_cols:
        if pd.api.types.is_float_dtype(df[c]) and df[c].dropna().mod(1).eq(0).all():
            df[c] = df[c].astype('Int64')
    return df


# ── helpers ──────────────────────────────────────────────────────────────────

def _bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        ps = p[ok]
        order = np.argsort(ps)
        ranked = ps[order] * ok.sum() / np.arange(1, ok.sum() + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q_ok = np.empty_like(ps)
        q_ok[order] = np.minimum(ranked, 1)
        q[ok] = q_ok
    return q


def _stars(q):
    return '' if not np.isfinite(q) else '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else ''


def overlap_stats(a, b):
    """Overlap of two boolean arrays over the same neurons, with Fisher's exact test."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n_ab, n_a, n_b, n = int((a & b).sum()), int(a.sum()), int(b.sum()), len(a)
    odds, p = fisher_exact([[n_ab, n_a - n_ab], [n_b - n_ab, n - n_a - n_b + n_ab]]) if n else (np.nan, np.nan)
    return dict(n_neurons=n, n_a=n_a, n_b=n_b, n_both=n_ab,
                p_b_given_a=n_ab / n_a if n_a else np.nan,
                p_a_given_b=n_ab / n_b if n_b else np.nan,
                p_b_overall=n_b / n if n else np.nan,
                jaccard=n_ab / (n_a + n_b - n_ab) if (n_a + n_b - n_ab) else np.nan,
                odds_ratio=odds, fisher_p=p)


def _effect_columns(results, method):
    sub = results[results['method'] == method]
    return [c for c in sub.columns if c not in RESERVED_COLS | set(ID_COLS)
            and pd.api.types.is_numeric_dtype(sub[c]) and sub[c].notna().any()]


def _wide(results, info, method, value='significant', id_cols=ID_COLS):
    """info joined with one column per test of method (value of that test)."""
    sub = results[results['method'] == method]
    wide = sub.pivot_table(index=list(id_cols), columns='test', values=value, aggfunc='first')
    return info.set_index(list(id_cols)).join(wide, how='inner')


class _Scopes:
    def __init__(self, info, scope_col, scopes):
        self.col = scope_col if scope_col in info.columns else None
        self.items = scopes if self.col else [('all', None)]

    def subset(self, df, value):
        return df if value is None or self.col is None else df[df[self.col] == value]


def _label(method, test):
    return f"{method} {test}"


def _safe(name):
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(name))


# ── 1. matrix: every test of method A x every test of method B ──────────────

def plot_overlap_matrix(results, info, method_a, method_b, output_folder, scopes, id_cols=ID_COLS):
    wa = _wide(results, info, method_a, id_cols=id_cols)
    wb = _wide(results, info, method_b, id_cols=id_cols)
    tests_a = [c for c in wa.columns if c not in info.columns]
    tests_b = [c for c in wb.columns if c not in info.columns]
    if not tests_a or not tests_b:
        return None
    both = wa.join(wb[tests_b], how='inner', lsuffix='__a', rsuffix='__b')
    col_a = {t: f'{t}__a' if f'{t}__a' in both.columns else t for t in tests_a}
    col_b = {t: f'{t}__b' if f'{t}__b' in both.columns else t for t in tests_b}

    records = []
    for (scope, value), ta, tb in itertools.product(scopes.items, tests_a, tests_b):
        sub = scopes.subset(both, value).dropna(subset=[col_a[ta], col_b[tb]])
        records.append(dict(scope=scope, test_a=ta, test_b=tb,
                            **overlap_stats(sub[col_a[ta]].astype(bool), sub[col_b[tb]].astype(bool))))
    stats = pd.DataFrame(records)
    stats['fisher_q'] = np.nan
    for scope, idx in stats.groupby('scope').groups.items():
        stats.loc[idx, 'fisher_q'] = _bh(stats.loc[idx, 'fisher_p'])
    stats['log2_odds'] = np.log2(stats['odds_ratio'].replace([0, np.inf], np.nan))
    names = (method_a, method_b) if method_a != method_b else (f'{method_a} (rows)', f'{method_b} (columns)')
    vq._save_table(stats.rename(columns={'test_a': names[0], 'test_b': names[1]}), output_folder,
                   'overlap_matrix_stats.txt')

    views = [('p_b_given_a', f'% of {method_a}-significant also {method_b}-significant', 'Purples', (0, 100), 100),
             ('p_a_given_b', f'% of {method_b}-significant also {method_a}-significant', 'Purples', (0, 100), 100),
             ('log2_odds', 'log2 odds ratio (Fisher; stars: BH-FDR over the matrix)', 'RdBu_r', None, 1)]
    for (scope, value), (col, title, cmap, lim, scale) in itertools.product(scopes.items, views):
        sub = stats[stats['scope'] == scope]
        mat = sub.pivot(index='test_a', columns='test_b', values=col) * scale
        q = sub.pivot(index='test_a', columns='test_b', values='fisher_q').reindex_like(mat)
        vals = mat.to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            continue
        if lim is None:
            v = np.nanmax(np.abs(vals))
            lim = (-v, v)
        fig, ax = plt.subplots(figsize=(0.42 * mat.shape[1] + 3, 0.4 * mat.shape[0] + 2))
        im = ax.imshow(vals, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect='auto')
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            if np.isfinite(vals[i, j]):
                txt = f"{vals[i, j]:.0f}" if scale == 100 else f"{vals[i, j]:.1f}"
                ax.text(j, i, txt + '\n' + _stars(q.iat[i, j]), ha='center', va='center', fontsize=4.5)
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=6)
        ax.set_xlabel(method_b, fontsize=7)
        ax.set_ylabel(method_a, fontsize=7)
        ax.set_title(f"{title}\n{scope}, {len(scopes.subset(both, value))} neurons", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.6).ax.tick_params(labelsize=6)
        vq.save_figure(fig, output_folder, f'overlap_matrix_{col}_{scope}')
        plt.close(fig)
    return stats


# ── 2. one test of A vs one test of B ────────────────────────────────────────

def _pair_table(results, info, a, b, id_cols=ID_COLS):
    def _one(method, test, tag):
        sub = results[(results['method'] == method) & (results['test'] == test)]
        cols = ['significant'] + _effect_columns(results, method)
        return sub[list(id_cols) + cols].rename(columns={c: f'{tag}:{c}' for c in cols})
    df = info.merge(_one(*a, 'a'), on=list(id_cols)).merge(_one(*b, 'b'), on=list(id_cols))
    sa, sb = df['a:significant'].astype(bool), df['b:significant'].astype(bool)
    la, lb = _label(*a), _label(*b)
    df['category'] = np.select([sa & sb, sa, sb], ['both', f'{la} only', f'{lb} only'], 'neither')
    return df


def _category_colors(a, b):
    return {'both': BOTH_COLOR, f'{_label(*a)} only': A_COLOR, f'{_label(*b)} only': B_COLOR,
            'neither': NEITHER_COLOR}


MIN_UNION_PER_MOUSE = 5  # mice with fewer neurons significant by either test are left out of the per-mouse panel


def _set_counts(sig_a, sig_b):
    """Set sizes of two boolean arrays and the overlap expected if the two
    tests picked neurons independently (same set sizes)."""
    sig_a, sig_b = np.asarray(sig_a, bool), np.asarray(sig_b, bool)
    n, n_a, n_b, n_ab = len(sig_a), int(sig_a.sum()), int(sig_b.sum()), int((sig_a & sig_b).sum())
    union = n_a + n_b - n_ab
    e_ab = n_a * n_b / n if n else np.nan
    e_union = n_a + n_b - e_ab
    return dict(n_neurons=n, n_a=n_a, n_b=n_b, n_both=n_ab, n_a_only=n_a - n_ab, n_b_only=n_b - n_ab,
                n_union=union,
                jaccard=n_ab / union if union else np.nan,
                jaccard_chance=e_ab / e_union if e_union else np.nan,
                pct_a_found_by_b=100 * n_ab / n_a if n_a else np.nan,
                pct_a_found_by_b_chance=100 * n_b / n if n else np.nan,
                pct_b_found_by_a=100 * n_ab / n_b if n_b else np.nan,
                pct_b_found_by_a_chance=100 * n_a / n if n else np.nan)


# Publication style (as glms_plotting_utils._plot_barplot_publication): Arial 7 pt,
# thin axes, text kept editable in the pdf.
PUB_RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
          'font.size': 7, 'axes.linewidth': 0.6, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
          'ytick.labelsize': 6.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
          'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
CHANCE_COLOR = '#a6a6a6'


def _pretty(name):
    return str(name).replace('_', ' ')


def _paired_obs_chance(ax, x0, obs, chance, color, p, dx=0.2):
    """Observed (colour) vs chance (grey) per mouse: one thin line per mouse,
    mean +/- SEM over mice, Wilcoxon stars over a bracket. Returns the top y."""
    xo, xc = x0 - dx, x0 + dx
    for o, c in zip(obs, chance):
        ax.plot([xo, xc], [o, c], color='#dcdcdc', lw=0.4, zorder=1)
    ax.scatter(np.full(len(obs), xo), obs, s=5, color=color, alpha=0.45, lw=0, zorder=2)
    ax.scatter(np.full(len(chance), xc), chance, s=5, color=CHANCE_COLOR, alpha=0.6, lw=0, zorder=2)
    for x, vals, c in ((xo, obs, color), (xc, chance, '#707070')):
        if len(vals):
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ax.errorbar(x, vals.mean(), yerr=se, fmt='o', ms=3.5, color=c, elinewidth=0.9, capsize=0, zorder=3)
    top = float(np.nanmax(np.r_[obs, chance])) if len(obs) else 0.0
    if np.isfinite(p):
        y = top * 1.06
        ax.plot([xo, xo, xc, xc], [y * 0.985, y, y, y * 0.985], color='#222222', lw=0.6)
        ax.text(x0, y, _stars(p) or 'n.s.', ha='center', va='bottom', fontsize=7 if _stars(p) else 6)
        top = y * 1.12
    return top


def _plot_pair_counts(df, a, b, out, scopes, min_union_per_mouse=MIN_UNION_PER_MOUSE):
    """Publication figure 'set_overlap': how different the two sets of
    significant neurons are, all mice pooled.

    a  Neurons significant by either test (= 100 %): A only / both / B only.
    b  Jaccard index (both / either) per mouse, observed vs chance.
    c  % of each test's set also significant by the other test, per mouse,
       observed vs chance.
    Chance = the overlap expected if the two tests picked neurons
    independently, given the mouse's set sizes and neuron count. Mean +/- SEM
    over mice (>= min_union_per_mouse neurons significant by either test);
    stars: Wilcoxon signed-rank observed vs chance. Numbers and tests are in
    set_overlap_pooled.txt (all / R+ / R-), set_overlap_per_mouse.txt and
    set_overlap_per_mouse_stats.txt.
    """
    from scipy.stats import wilcoxon

    la, lb = _label(*a), _label(*b)
    sa, sb = df['a:significant'].astype(bool), df['b:significant'].astype(bool)

    pooled_rows = []
    for scope, value in scopes.items:
        idx = scopes.subset(df, value).index
        fisher = overlap_stats(sa[idx], sb[idx])
        pooled_rows.append(dict(scope=scope, **_set_counts(sa[idx], sb[idx]),
                                odds_ratio=fisher['odds_ratio'], fisher_p=fisher['fisher_p']))
    pooled = pd.DataFrame(pooled_rows)
    pooled.insert(1, 'test_a', la)
    pooled.insert(2, 'test_b', lb)
    vq._save_table(pooled, out, 'set_overlap_pooled.txt')

    mouse_rows = []
    for mouse, m in df.groupby('mouse_id'):
        rec = _set_counts(m['a:significant'].astype(bool), m['b:significant'].astype(bool))
        group = m[scopes.col].iloc[0] if scopes.col else None
        mouse_rows.append(dict(mouse_id=mouse, group=group, **rec))
    per_mouse = pd.DataFrame(mouse_rows)
    vq._save_table(per_mouse, out, 'set_overlap_per_mouse.txt')
    kept = per_mouse[per_mouse['n_union'] >= min_union_per_mouse]

    stats_rows, per_metric = [], {}
    for col in ('jaccard', 'pct_a_found_by_b', 'pct_b_found_by_a'):
        sub = kept.dropna(subset=[col, f'{col}_chance'])
        obs, chance = sub[col].to_numpy(float), sub[f'{col}_chance'].to_numpy(float)
        p = wilcoxon(obs, chance)[1] if len(obs) >= 5 and np.any(obs != chance) else np.nan
        per_metric[col] = (obs, chance, p)
        stats_rows.append(dict(test_a=la, test_b=lb, metric=col, n_mice=len(obs),
                               mean_observed=obs.mean() if len(obs) else np.nan,
                               sem_observed=obs.std(ddof=1) / np.sqrt(len(obs)) if len(obs) > 1 else np.nan,
                               mean_chance=chance.mean() if len(obs) else np.nan,
                               median_observed=np.median(obs) if len(obs) else np.nan, wilcoxon_p=p))

    # Names on the figure: the methods, or the tests when both come from one method.
    name_a, name_b = (a[0], b[0]) if a[0] != b[0] else (_pretty(a[1]), _pretty(b[1]))
    tot = pooled.iloc[0]  # first scope = all neurons

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(5.4, 1.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.8, 0.85, 1.3], wspace=0.8)

        # a: share of the neurons significant by either test
        ax = fig.add_subplot(gs[0, 0])
        segs = [(int(tot['n_a_only']), f'{name_a} only', A_COLOR), (int(tot['n_both']), 'Both', BOTH_COLOR),
                (int(tot['n_b_only']), f'{name_b} only', B_COLOR)]
        left = 0.0
        for i, (n_seg, label, color) in enumerate(segs):
            pct = 100 * n_seg / tot['n_union'] if tot['n_union'] else 0
            ax.barh(0, pct, left=left, height=0.5, color=color, edgecolor='white', linewidth=0.8)
            if pct >= 8:
                ax.text(left + pct / 2, 0, f"{pct:.0f}%", ha='center', va='center', color='white', fontsize=7)
            # Labels alternate above / below the bar so neighbours never collide.
            above = i != 1
            ax.text(left + pct / 2, 0.33 if above else -0.33, f"{label}\n{n_seg}", ha='center',
                    va='bottom' if above else 'top', color=color, fontsize=6, linespacing=1.1)
            left += pct
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_xlabel(f"% of neurons significant by either\n"
                      f"({int(tot['n_union'])} of {int(tot['n_neurons'])} neurons)")

        # b: Jaccard index per mouse
        ax = fig.add_subplot(gs[0, 1])
        obs, chance, p = per_metric['jaccard']
        top = _paired_obs_chance(ax, 0, obs, chance, BOTH_COLOR, p, dx=0.3)
        ax.set_xticks([])  # observed (left) vs chance (right): see the legend of panel c
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(0, max(top, 0.05))
        ax.set_ylabel('Jaccard index')
        ax.set_xlabel(f"{len(obs)} mice")
        ax.spines[['top', 'right']].set_visible(False)

        # c: % of each set also significant by the other test
        ax = fig.add_subplot(gs[0, 2])
        top = 0.0
        for x0, col, color in ((0, 'pct_a_found_by_b', A_COLOR), (1, 'pct_b_found_by_a', B_COLOR)):
            obs, chance, p = per_metric[col]
            top = max(top, _paired_obs_chance(ax, x0, obs, chance, color, p))
        ax.set_xticks([0, 1], [f"{name_a} set", f"{name_b} set"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, min(max(top, 5), 115))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel('% also significant\nby the other test')
        ax.spines[['top', 'right']].set_visible(False)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker='o', ls='', ms=3.5, color='#444444', label='Observed'),
                           Line2D([], [], marker='o', ls='', ms=3.5, color=CHANCE_COLOR, label='Chance')],
                  frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.2)
        vq.save_figure(fig, out, 'set_overlap', file_formats=('png', 'pdf', 'svg'))
        plt.close(fig)

    stats = pd.DataFrame(stats_rows)
    if not stats.empty:
        stats['wilcoxon_q'] = _bh(stats['wilcoxon_p'])
        vq._save_table(stats, out, 'set_overlap_per_mouse_stats.txt')


def _plot_pair_area_groups(df, a, b, out, scopes, area_col, area_order, area_colors, min_neurons):
    if area_col not in df.columns or df[area_col].isna().all():
        return
    records = []
    for (scope, value), (area, sub) in itertools.product(scopes.items, df.groupby(area_col)):
        sub = scopes.subset(sub, value)
        if len(sub) >= min_neurons:
            records.append(dict(scope=scope, area=area,
                                **overlap_stats(sub['a:significant'].astype(bool), sub['b:significant'].astype(bool))))
    stats = pd.DataFrame(records)
    if stats.empty:
        return
    stats['fisher_q'] = np.nan
    for scope, idx in stats.groupby('scope').groups.items():
        stats.loc[idx, 'fisher_q'] = _bh(stats.loc[idx, 'fisher_p'])
    vq._save_table(stats, out, 'per_area_stats.txt')
    order = [g for g in area_order if g in set(stats['area'])] + sorted(set(stats['area']) - set(area_order))
    la, lb = _label(*a), _label(*b)
    for scope, _ in scopes.items:
        sub = stats[stats['scope'] == scope].set_index('area').reindex(order).dropna(subset=['n_neurons'])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(0.45 * len(sub) + 1.8, 2.8))
        x, w = np.arange(len(sub)), 0.27
        bars = [('p_b_overall', f'{lb}-sig, all neurons', '#d9d9d9', None),
                ('p_b_given_a', f'{lb}-sig among {la}-sig', BOTH_COLOR, None),
                ('p_a_given_b', f'{la}-sig among {lb}-sig', 'white', '///')]
        for k, (col, label, color, hatch) in enumerate(bars):
            ax.bar(x + (k - 1) * w, sub[col] * 100, width=w, label=label, color=color, hatch=hatch,
                   edgecolor=BOTH_COLOR if hatch else 'none', linewidth=0.6)
        for xi, q in zip(x, sub['fisher_q']):
            ax.text(xi, 102, _stars(q), ha='center', fontsize=7)
        ax.set_xticks(x, sub.index, rotation=60, ha='right', fontsize=6)
        for tick, area in zip(ax.get_xticklabels(), sub.index):
            tick.set_color((area_colors or {}).get(area, 'black'))
        ax.set_ylim(0, 108)
        ax.set_ylabel('% of neurons', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=5.5, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title(f"{la} vs {lb} - {scope}\n(stars: Fisher, BH-FDR over areas; >= {min_neurons} neurons)",
                     fontsize=7)
        vq.save_figure(fig, out, f'per_area_{scope}')
        plt.close(fig)


def _box(ax, data, labels, colors, ylabel):
    data = [d.dropna() for d in data]
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
    ax.set_xticks([1, 2], [f"{l}\nn={len(d)}" for l, d in zip(labels, data)], fontsize=5)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.tick_params(labelsize=5)
    ax.spines[['top', 'right']].set_visible(False)
    return mannwhitneyu(*data)[1] if all(len(d) for d in data) else np.nan


def _plot_pair_effect_sizes(df, a, b, out, scopes, effects_a, effects_b):
    colors = _category_colors(a, b)
    la, lb = _label(*a), _label(*b)
    records = []
    for ea, eb in itertools.product(effects_a, effects_b):
        xa, yb = f'a:{ea}', f'b:{eb}'
        n_s = len(scopes.items)
        fig, axes = plt.subplots(3, n_s, figsize=(2.6 * n_s, 7.0), squeeze=False,
                                 gridspec_kw=dict(hspace=0.9, wspace=0.45, height_ratios=(1.5, 1, 1)))
        for c, (scope, value) in enumerate(scopes.items):
            sub = scopes.subset(df, value).dropna(subset=[xa, yb])
            ax = axes[0, c]
            for cat in ['neither', f'{lb} only', f'{la} only', 'both']:
                s = sub[sub['category'] == cat]
                ax.scatter(s[xa], s[yb], s=3, alpha=0.5, lw=0, color=colors[cat], label=f"{cat} ({len(s)})",
                           rasterized=True)
            rho, p = spearmanr(sub[xa], sub[yb]) if len(sub) > 2 else (np.nan, np.nan)
            per_mouse = pd.Series([spearmanr(m[xa], m[yb])[0] for _, m in sub.groupby('mouse_id') if len(m) >= 10],
                                  dtype=float).dropna()
            ax.axvline(0, color='#999999', lw=0.5, ls='--')
            ax.set_title(f"{scope}: Spearman ρ={rho:.2f} (p={p:.1e})\nmedian ρ per mouse="
                         f"{per_mouse.median():.2f} ({len(per_mouse)} mice)", fontsize=6)
            ax.set_xlabel(f"{la}: {ea}", fontsize=6)
            ax.set_ylabel(f"{lb}: {eb}", fontsize=6)
            ax.tick_params(labelsize=5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=4.5, frameon=False, markerscale=2)

            sig_a, sig_b = sub['a:significant'].astype(bool), sub['b:significant'].astype(bool)
            p_a = _box(axes[1, c], [sub.loc[sig_b, xa], sub.loc[~sig_b, xa]], [f'{b[0]} sig', f'{b[0]} n.s.'],
                       [B_COLOR, NEITHER_COLOR], f"{la}: {ea}")
            axes[1, c].set_title(f"{la} effect\nby {lb} significance\nMann-Whitney p={p_a:.1e}", fontsize=5.5)
            p_b = _box(axes[2, c], [sub.loc[sig_a, yb], sub.loc[~sig_a, yb]], [f'{a[0]} sig', f'{a[0]} n.s.'],
                       [A_COLOR, NEITHER_COLOR], f"{lb}: {eb}")
            axes[2, c].set_title(f"{lb} effect\nby {la} significance\nMann-Whitney p={p_b:.1e}", fontsize=5.5)
            base = dict(test_a=la, test_b=lb, effect_a=ea, effect_b=eb, scope=scope, n_neurons=len(sub),
                        spearman_rho=rho, spearman_p=p, median_rho_per_mouse=per_mouse.median(),
                        n_mice_rho=len(per_mouse))
            records += [dict(base, comparison=f"{ea} by {lb} significance",
                             median_sig=sub.loc[sig_b, xa].median(), median_ns=sub.loc[~sig_b, xa].median(),
                             mannwhitney_p=p_a),
                        dict(base, comparison=f"{eb} by {la} significance",
                             median_sig=sub.loc[sig_a, yb].median(), median_ns=sub.loc[~sig_a, yb].median(),
                             mannwhitney_p=p_b)]
        fig.suptitle(f"Effect sizes: {la} ({ea}) vs {lb} ({eb})", fontsize=8)
        vq.save_figure(fig, out, f'effect_size_{_safe(ea)}__vs__{_safe(eb)}')
        plt.close(fig)
    return records


# ── PSTH of each set (both / A only / B only / neither) ──────────────────────

# PETH caches of mh_ephys_analysis/scripts/psth_per_mouse_with_barplots.py:
# {PETH_CACHE_ROOT}/{filter-set folder}/{PETH_CACHE_FILE}, one whisker_0 session per mouse.
PETH_CACHE_ROOT = os.path.join(vq._MYRIAM_RESULTS, 'passive', '5ms')
PETH_CACHE_FILE = 'all_peth_rows_hit_vs_miss.pkl'
PETH_CONDITION_STYLES = {'hit': '-', 'miss': '--'}
PETH_PANEL_LABELS = {'whisker_trial': 'Whisker trials', 'auditory_trial': 'Auditory trials'}
MIN_UNITS_PER_MOUSE_PSTH = 3  # a mouse enters a set's average with at least this many of its neurons


def load_peth_cache(path, mice=None):
    """PETH cache (list of per-session DataFrames) -> dict(table, time):
    table has mouse_id, neuron_id (= the cache's unit_id), panel (trial_type),
    condition (context), baseline_fr and peth (one array per row). Only
    (neuron, trial type) with enough trials in every condition are kept."""
    import pickle
    with open(vq._long_path(path), 'rb') as f:
        rows = pickle.load(f)
    df = pd.concat([r for r in rows if not r.empty], ignore_index=True)
    if mice is not None:
        df = df[df['mouse_id'].isin(set(mice))]
    time = np.asarray(df['time'].iloc[0], dtype=float)
    ok = df.groupby(['mouse_id', 'unit_id', 'trial_type'])['min_trials_ok'].transform('all')
    df = df[ok.astype(bool)]
    keep = ['mouse_id', 'unit_id', 'trial_type', 'context', 'baseline_fr', 'peth'] + \
           [c for c in ('peth_sem', 'n_trials') if c in df.columns]
    table = df[keep].rename(
        columns={'unit_id': 'neuron_id', 'trial_type': 'panel', 'context': 'condition'}).reset_index(drop=True)
    table['neuron_id'] = table['neuron_id'].astype(int)
    print(f"[overlap] PETH cache: {table[['mouse_id', 'neuron_id']].drop_duplicates().shape[0]} neurons, "
          f"{table['mouse_id'].nunique()} mice, conditions {sorted(table['condition'].unique())}, "
          f"{time[0]:.3f} to {time[-1]:.3f} s")
    return dict(table=table, time=time)


def _mouse_mean_traces(table, baseline_correct):
    """Per mouse x panel x condition: mean PETH over the mouse's neurons."""
    mat = np.stack(table['peth'].to_numpy()).astype(float)
    if baseline_correct:
        bl = table['baseline_fr'].to_numpy(dtype=float)
        mat = mat - np.where(np.isfinite(bl), bl, np.nanmean(mat[:, :max(1, mat.shape[1] // 4)], axis=1))[:, None]
    out = {}
    for key, idx in table.groupby(['mouse_id', 'panel', 'condition']).indices.items():
        out[key] = (np.nanmean(mat[idx], axis=0), len(idx))
    return out


def _plot_pair_psths(df, a, b, out, scopes, peth, min_units=MIN_UNITS_PER_MOUSE_PSTH, id_cols=ID_COLS):
    """PSTH of the four sets of the pair (both / A only / B only / neither),
    mean over mice (each mouse = mean of its neurons in the set, >= min_units
    neurons) +/- SEM over mice, one line per condition (hit solid, miss
    dashed). Two figures per scope: psth_by_set (one column per set) and
    psth_sets_overlaid (sets overlaid, one column per condition); each
    baseline-corrected (delta FR from the pre-stimulus baseline) and raw."""
    colors = _category_colors(a, b)
    cats = list(colors)  # both, A only, B only, neither
    time = peth['time'] * 1000
    table = peth['table'].merge(df[list(id_cols) + ['category'] + ([scopes.col] if scopes.col else [])],
                                on=list(id_cols))
    if table.empty:
        print(f"[overlap] PSTH: no PETH for the neurons of {_label(*a)} vs {_label(*b)}.")
        return
    panels = [p for p in PETH_PANEL_LABELS if p in set(table['panel'])] + \
             sorted(set(table['panel']) - set(PETH_PANEL_LABELS))
    conditions = [c for c in PETH_CONDITION_STYLES if c in set(table['condition'])] + \
                 sorted(set(table['condition']) - set(PETH_CONDITION_STYLES))
    styles = {c: PETH_CONDITION_STYLES.get(c, ':') for c in conditions}
    count_rows = []

    for (scope, value), baseline_correct in itertools.product(scopes.items, (True, False)):
        sub = table if value is None else table[table[scopes.col] == value]
        traces, n_units = {}, {}
        for cat in cats:
            cs = sub[sub['category'] == cat]
            per_neuron = cs.drop_duplicates(list(id_cols) + ['panel'])
            units_per_mouse = per_neuron.groupby(['mouse_id', 'panel']).size()
            keep = units_per_mouse[units_per_mouse >= min_units].index
            cs = cs.set_index(['mouse_id', 'panel']).loc[lambda d: d.index.isin(keep)].reset_index()
            mm = _mouse_mean_traces(cs, baseline_correct) if not cs.empty else {}
            for panel, cond in itertools.product(panels, conditions):
                stack = np.array([v[0] for (m, p, c), v in mm.items() if p == panel and c == cond])
                traces[(cat, panel, cond)] = stack
                n_units[(cat, panel)] = (int(cs.loc[cs['panel'] == panel, list(id_cols)].drop_duplicates().shape[0]),
                                         int(cs.loc[cs['panel'] == panel, 'mouse_id'].nunique()))
            if baseline_correct:
                for panel in panels:
                    count_rows.append(dict(scope=scope, set=cat, panel=panel,
                                           n_neurons=n_units[(cat, panel)][0], n_mice=n_units[(cat, panel)][1]))
        ylabel = 'Δ firing rate from baseline (Hz)' if baseline_correct else 'Firing rate (Hz)'
        tag = '' if baseline_correct else '_raw'

        def _draw(ax, stack, color, style, label=None):
            if len(stack) == 0:
                return
            m = np.nanmean(stack, axis=0)
            se = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(len(stack)) if len(stack) > 1 else np.zeros_like(m)
            ax.plot(time, m, color=color, ls=style, lw=1.0, label=label)
            ax.fill_between(time, m - se, m + se, color=color, alpha=0.18, lw=0)

        def _style_ax(ax):
            ax.axvline(0, color='#9e9e9e', lw=0.6, ls=':')
            if baseline_correct:
                ax.axhline(0, color='#cccccc', lw=0.5)
            ax.tick_params(labelsize=5.5, length=2)
            ax.spines[['top', 'right']].set_visible(False)

        # One column per set.
        fig, axes = plt.subplots(len(panels), len(cats), figsize=(1.6 * len(cats) + 0.6, 1.45 * len(panels) + 0.6),
                                 squeeze=False, sharex=True, sharey=True,
                                 gridspec_kw=dict(wspace=0.15, hspace=0.45))
        for (r, panel), (c, cat) in itertools.product(enumerate(panels), enumerate(cats)):
            ax = axes[r, c]
            for cond in conditions:
                _draw(ax, traces[(cat, panel, cond)], colors[cat], styles[cond], cond if (r, c) == (0, 0) else None)
            n_n, n_m = n_units[(cat, panel)]
            ax.set_title((f"{cat}\n" if r == 0 else '') + f"{n_n} neurons, {n_m} mice", fontsize=5.5,
                         color=colors[cat] if r == 0 else 'black')
            _style_ax(ax)
            if c == 0:
                ax.set_ylabel(f"{PETH_PANEL_LABELS.get(panel, panel)}\n{ylabel}", fontsize=5.5)
            if r == len(panels) - 1:
                ax.set_xlabel('Time from stimulus (ms)', fontsize=5.5)
        axes[0, 0].legend(fontsize=5, frameon=False, handlelength=1.8, loc='upper left')
        fig.suptitle(f"{_label(*a)} vs {_label(*b)} - {scope} (mean ± SEM over mice)", fontsize=7, y=1.02)
        vq.save_figure(fig, os.path.join(out, 'psth'), f'psth_by_set_{scope}{tag}')
        plt.close(fig)

        # Sets overlaid: one column per condition.
        fig, axes = plt.subplots(len(panels), len(conditions),
                                 figsize=(1.9 * len(conditions) + 1.6, 1.45 * len(panels) + 0.6),
                                 squeeze=False, sharex=True, sharey=True, gridspec_kw=dict(wspace=0.15, hspace=0.45))
        for (r, panel), (c, cond) in itertools.product(enumerate(panels), enumerate(conditions)):
            ax = axes[r, c]
            for cat in cats:
                n_n, n_m = n_units[(cat, panel)]
                _draw(ax, traces[(cat, panel, cond)], colors[cat], '-', f"{cat} ({n_n} / {n_m} mice)")
            ax.set_title(f"{PETH_PANEL_LABELS.get(panel, panel)} - {cond}", fontsize=6)
            _style_ax(ax)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=5.5)
            if r == len(panels) - 1:
                ax.set_xlabel('Time from stimulus (ms)', fontsize=5.5)
            if c == len(conditions) - 1:
                ax.legend(fontsize=4.8, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), handlelength=1.2)
        fig.suptitle(f"{_label(*a)} vs {_label(*b)} - {scope} (mean ± SEM over mice)", fontsize=7, y=1.02)
        vq.save_figure(fig, os.path.join(out, 'psth'), f'psth_sets_overlaid_{scope}{tag}')
        plt.close(fig)
    vq._save_table(pd.DataFrame(count_rows), os.path.join(out, 'psth'), 'psth_counts.txt')


# Stimulation artifact: an isolated jump in the first post-stimulus bin (0-5 ms) of the
# whisker-trial PETH, on hits AND misses - too early for a neural response, and found
# in areas with no reason to respond within 5 ms (ALM, CA3, ORB, hypothalamus, ...).
ARTIFACT_TRIAL_TYPE = 'whisker_trial'
ARTIFACT_MIN_JUMP_HZ = 40      # first-bin rate minus the pre-stimulus mean
ARTIFACT_MAX_LATER_FRACTION = 1 / 3  # max over the next 25 ms must stay below this fraction of the jump


def stim_artifact_units(peth, trial_type=ARTIFACT_TRIAL_TYPE, min_jump_hz=ARTIFACT_MIN_JUMP_HZ,
                        max_later_fraction=ARTIFACT_MAX_LATER_FRACTION, conditions=('hit', 'miss'),
                        id_cols=ID_COLS):
    """Neurons whose trial_type PETH jumps in the first post-stimulus bin and
    falls back right after, in every condition (default hit and miss).

    jump = first-bin rate - mean rate of the bins up to t = 0; later = max over
    the next 25 ms - same mean. Flagged when jump >= min_jump_hz and
    later <= max_later_fraction * jump (floored at 1 Hz). Time labels are the
    bins' right edges, so the first post-stimulus bin is the first label > 0.
    Returns a DataFrame with id_cols and the jump per condition."""
    t = np.round(peth['time'] * 1000, 6)
    first = int(np.argmax(t > 0))
    bin_ms = t[1] - t[0]
    later = (t > t[first]) & (t <= t[first] + 25 + 1e-9)
    pre = t <= 0
    table = peth['table']
    sub = table[(table['panel'] == trial_type) & table['condition'].isin(conditions)]
    if sub.empty:
        return pd.DataFrame(columns=list(id_cols))
    mat = np.stack(sub['peth'].to_numpy()).astype(float)
    base = mat[:, pre].mean(1)
    jump = mat[:, first] - base
    rest = mat[:, later].max(1) - base
    flag = np.isfinite(jump) & (jump >= min_jump_hz) & (np.maximum(rest, 1) <= max_later_fraction * jump)
    d = sub[list(id_cols) + ['condition']].assign(jump_hz=jump, flag=flag)
    wide = d.pivot_table(index=list(id_cols), columns='condition', values=['flag', 'jump_hz'], aggfunc='first')
    all_flag = wide['flag'].reindex(columns=list(conditions)).fillna(False).astype(bool).all(axis=1)
    out = wide['jump_hz'].reindex(columns=list(conditions)).add_prefix('jump_hz_')[all_flag].reset_index()
    print(f"[overlap] Stim artifact ({trial_type}, first {bin_ms:g} ms bin >= {min_jump_hz} Hz above baseline, "
          f"isolated, in {'/'.join(conditions)}): {len(out)} of {len(wide)} neurons.")
    return out


N_EXAMPLE_NEURONS = 10
EXAMPLE_PANEL_COLORS = {'whisker_trial': None, 'auditory_trial': '#7f7f7f'}  # None = the set's colour


def _plot_pair_examples(df, a, b, out, peth, n_examples=N_EXAMPLE_NEURONS, seed=0, id_cols=ID_COLS):
    """Example neurons of each set (both / A only / B only / neither):
    n_examples drawn at random (fixed seed) among the set's neurons with a
    PETH. One panel per neuron: its PETH per trial type (whisker in the set's
    colour, auditory in grey) and condition (hit solid, miss dashed), raw
    firing rate, shaded +/- SEM over trials when the cache has it. Title:
    mouse, neuron, area and the first effect size of each test."""
    colors = _category_colors(a, b)
    time = peth['time'] * 1000
    table = peth['table']
    has_peth = table[list(id_cols)].drop_duplicates()
    pool = df.merge(has_peth, on=list(id_cols))
    eff_a = next((c for c in df.columns if c.startswith('a:') and c != 'a:significant'), None)
    eff_b = next((c for c in df.columns if c.startswith('b:') and c != 'b:significant'), None)
    area_col = next((c for c in ('area_acronym_custom', 'area_group') if c in df.columns), None)
    by_neuron = table.groupby(list(id_cols)).indices
    rng = np.random.default_rng(seed)
    chosen_rows = []
    for cat, color in colors.items():
        cands = pool[pool['category'] == cat]
        if cands.empty:
            continue
        pick = cands.iloc[rng.choice(len(cands), size=min(n_examples, len(cands)), replace=False)]
        n_cols = 5
        n_rows = int(np.ceil(len(pick) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.55 * n_cols + 0.4, 1.35 * n_rows + 0.7),
                                 squeeze=False, sharex=True, gridspec_kw=dict(wspace=0.45, hspace=0.75))
        for ax in axes.ravel()[len(pick):]:
            ax.axis('off')
        # Dicts, not itertuples: column names like 'a:delta_test_corr' aren't valid attributes.
        for ax, row in zip(axes.ravel(), pick.to_dict('records')):
            rows = table.iloc[by_neuron[tuple(row[c] for c in id_cols)]]
            for r in rows.itertuples(index=False):
                pc = EXAMPLE_PANEL_COLORS.get(r.panel) or color
                m = np.asarray(r.peth, dtype=float)
                ax.plot(time, m, color=pc, ls=PETH_CONDITION_STYLES.get(r.condition, ':'), lw=0.8)
                if 'peth_sem' in rows.columns and r.peth_sem is not None:
                    se = np.asarray(r.peth_sem, dtype=float)
                    ax.fill_between(time, m - se, m + se, color=pc, alpha=0.15, lw=0)
            ax.axvline(0, color='#9e9e9e', lw=0.5, ls=':')
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=4.5, length=1.5)
            ax.spines[['top', 'right']].set_visible(False)
            parts = [f"{row['mouse_id']} #{row['neuron_id']}" + (f" {row[area_col]}" if area_col else '')]
            vals = [f"{lab}={row[col]:.2f}" for lab, col in ((a[0], eff_a), (b[0], eff_b))
                    if col and pd.notna(row.get(col))]
            ax.set_title(parts[0] + ('\n' + ', '.join(vals) if vals else ''), fontsize=4.5)
            chosen_rows.append(dict(set=cat, **{c: row[c] for c in id_cols},
                                    area=row[area_col] if area_col else None,
                                    **({eff_a: row[eff_a]} if eff_a else {}), **({eff_b: row[eff_b]} if eff_b else {})))
        for ax in axes[-1]:
            ax.set_xlabel('Time (ms)', fontsize=5)
        for ax in axes[:, 0]:
            ax.set_ylabel('FR (Hz)', fontsize=5)
        effects = ', '.join(f"{m}: {c[2:]}" for m, c in ((a[0], eff_a), (b[0], eff_b)) if c)
        fig.suptitle(f"{cat}: {len(pick)} random of {len(cands)} neurons ({_label(*a)} vs {_label(*b)})\n"
                     f"whisker = colour, auditory = grey; hit solid, miss dashed; ± SEM over trials"
                     + (f"; {effects}" if effects else ''), fontsize=6, y=1.02, color=color)
        vq.save_figure(fig, os.path.join(out, 'psth', 'examples'), f"examples_{_safe(cat)}")
        plt.close(fig)
    if chosen_rows:
        vq._save_table(pd.DataFrame(chosen_rows), os.path.join(out, 'psth', 'examples'), 'example_neurons.txt')


def compare_pairs(results, info, pairs, output_folder, scopes, area_col='area_group', area_order=(),
                  area_colors=None, min_neurons_area=MIN_NEURONS_AREA, effects=None, id_cols=ID_COLS, peth=None):
    """pairs: [((method_a, test_a), (method_b, test_b)), ...]. effects:
    {method: [effect columns]} (default: every effect column of the method).
    peth: load_peth_cache() output, for the PSTH of each set (optional)."""
    available = set(map(tuple, results[['method', 'test']].drop_duplicates().to_numpy()))
    count_records, effect_records = [], []
    for a, b in pairs:
        a, b = tuple(a), tuple(b)
        missing = [x for x in (a, b) if x not in available]
        if missing:
            print(f"[overlap] Pair {_label(*a)} / {_label(*b)}: {missing} not in the results. Skipped.")
            continue
        df = _pair_table(results, info, a, b, id_cols)
        if df.empty:
            continue
        out = os.path.join(output_folder, 'pairs', f"{_safe(a[0])}-{_safe(a[1])}__vs__{_safe(b[0])}-{_safe(b[1])}")
        vq._save_table(df, out, 'neurons.txt')
        _plot_pair_counts(df, a, b, out, scopes)
        _plot_pair_area_groups(df, a, b, out, scopes, area_col, list(area_order), area_colors, min_neurons_area)
        eff_a = (effects or {}).get(a[0], _effect_columns(results, a[0]))
        eff_b = (effects or {}).get(b[0], _effect_columns(results, b[0]))
        eff_a = [e for e in eff_a if f'a:{e}' in df.columns and df[f'a:{e}'].notna().any()]
        eff_b = [e for e in eff_b if f'b:{e}' in df.columns and df[f'b:{e}'].notna().any()]
        effect_records += _plot_pair_effect_sizes(df, a, b, out, scopes, eff_a, eff_b)
        if peth is not None:
            _plot_pair_psths(df, a, b, out, scopes, peth, id_cols=id_cols)
            _plot_pair_examples(df, a, b, out, peth, id_cols=id_cols)
        for scope, value in scopes.items:
            sub = scopes.subset(df, value)
            count_records.append(dict(test_a=_label(*a), test_b=_label(*b), scope=scope,
                                      **overlap_stats(sub['a:significant'].astype(bool),
                                                      sub['b:significant'].astype(bool))))
        print(f"[overlap] {_label(*a)} vs {_label(*b)}: {len(df)} neurons -> {out}")
    if count_records:
        counts = pd.DataFrame(count_records)
        counts['fisher_q'] = _bh(counts['fisher_p'])
        vq._save_table(counts, output_folder, 'pairs_overlap_stats.txt')
    if effect_records:
        eff = pd.DataFrame(effect_records)
        eff['mannwhitney_q'] = _bh(eff['mannwhitney_p'])
        vq._save_table(eff, output_folder, 'pairs_effect_size_stats.txt')


# ── 3. intersections of several sets (UpSet style) ───────────────────────────

def plot_upset(results, info, sets, output_folder, scopes, max_bars=25, id_cols=ID_COLS):
    """sets: {label: (method, test)}. Neurons with a result for every set."""
    table = info.set_index(list(id_cols))
    labels = []
    for label, (method, test) in sets.items():
        s = results[(results['method'] == method) & (results['test'] == test)].set_index(list(id_cols))
        if s.empty:
            print(f"[overlap] UpSet: {label} ({method} {test}) not in the results. Left out.")
            continue
        table = table.join(s['significant'].rename(label), how='inner')
        labels.append(label)
    if len(labels) < 2:
        return
    methods = list(dict.fromkeys(sets[l][0] for l in labels))
    palette = dict(zip(methods, [A_COLOR, B_COLOR, BOTH_COLOR, '#33a02c', '#b15928']))
    for scope, value in scopes.items:
        sub = scopes.subset(table, value)
        if sub.empty:
            continue
        flags = sub[labels].astype(bool).to_numpy()
        none = tuple([False] * len(labels))
        combos = pd.Series([tuple(r) for r in flags]).value_counts()
        n_none = int(combos.get(none, 0))
        combos = combos.drop(none, errors='ignore').head(max_bars)
        vq._save_table(pd.DataFrame([dict(zip(labels, c), n_neurons=n) for c, n in combos.items()]),
                       output_folder, f'upset_{scope}.txt')
        fig, (ax_bar, ax_dot) = plt.subplots(2, 1, figsize=(0.28 * len(combos) + 3.0, 1.8 + 0.3 * len(labels) + 1.5),
                                             sharex=True, gridspec_kw=dict(height_ratios=(1.6, 1), hspace=0.05))
        x = np.arange(len(combos))
        ax_bar.bar(x, combos.to_numpy(), color='#525252', width=0.7)
        for xi, v in zip(x, combos.to_numpy()):
            ax_bar.text(xi, v, str(v), ha='center', va='bottom', fontsize=4.5)
        ax_bar.set_ylabel('neurons', fontsize=7)
        ax_bar.tick_params(labelsize=6)
        ax_bar.spines[['top', 'right']].set_visible(False)
        ax_bar.set_title(f"{scope}: {len(sub)} neurons, {n_none} in no set", fontsize=7)
        for xi, combo in zip(x, combos.index):
            on = [i for i, v in enumerate(combo) if v]
            ax_dot.scatter([xi] * len(labels), range(len(labels)), s=12, color='#d9d9d9', zorder=1)
            ax_dot.scatter([xi] * len(on), on, s=12, zorder=2, color=[palette[sets[labels[i]][0]] for i in on])
            if len(on) > 1:
                ax_dot.plot([xi, xi], [min(on), max(on)], color='#525252', lw=1, zorder=1)
        ax_dot.set_yticks(range(len(labels)), [f"{l} ({int(flags[:, i].sum())})" for i, l in enumerate(labels)],
                          fontsize=6)
        ax_dot.set_ylim(len(labels) - 0.4, -0.6)
        ax_dot.set_xticks([])
        for s in ('top', 'right', 'bottom'):
            ax_dot.spines[s].set_visible(False)
        vq.save_figure(fig, output_folder, f'upset_{scope}')
        plt.close(fig)


# ── entry points ─────────────────────────────────────────────────────────────

def run_overlap(results, info, output_folder, pairs=(), upset_sets=None, matrices=None, effects=None,
                scope_col='reward_group', scopes=REWARD_SCOPES, area_col='area_group', area_order=(),
                area_colors=None, min_neurons_area=MIN_NEURONS_AREA, id_cols=ID_COLS, peth=None,
                exclude_units=None):
    """All outputs (see the module docstring).

    matrices: [(method_a, method_b), ...]; default every pair of distinct
    methods in results. pairs, upset_sets: see compare_pairs / plot_upset.
    peth: load_peth_cache() output; adds pairs/<pair>/psth/ (PSTH of each set).
    exclude_units: DataFrame with id_cols of neurons left out of everything
    (e.g. stim_artifact_units()); saved as excluded_units.txt.
    """
    id_cols = list(id_cols)
    results = _normalise_ids(results, id_cols)
    info = _normalise_ids(info, id_cols).drop_duplicates(id_cols)
    if exclude_units is not None and len(exclude_units):
        excl = _normalise_ids(exclude_units, id_cols)
        dropped = info.merge(excl[id_cols].drop_duplicates(), on=id_cols)
        info = info.merge(excl[id_cols].drop_duplicates(), on=id_cols, how='left', indicator=True)
        info = info[info['_merge'] == 'left_only'].drop(columns='_merge').reset_index(drop=True)
        vq._save_table(excl.merge(dropped[id_cols], on=id_cols), output_folder, 'excluded_units.txt')
        print(f"[overlap] Excluded {len(dropped)} neurons ({len(excl)} in the exclusion list).")
    scopes = _Scopes(info, scope_col, scopes)
    methods = list(dict.fromkeys(results['method']))
    print(f"[overlap] {len(info)} neurons; methods: "
          + ', '.join(f"{m} ({results.loc[results['method'] == m, 'test'].nunique()} tests)" for m in methods))
    if matrices is None:
        matrices = list(itertools.combinations(methods, 2))
    for ma, mb in matrices:
        plot_overlap_matrix(results, info, ma, mb, os.path.join(output_folder, f'matrix_{_safe(ma)}_vs_{_safe(mb)}'),
                            scopes, id_cols)
    if pairs:
        compare_pairs(results, info, pairs, output_folder, scopes, area_col, area_order, area_colors,
                      min_neurons_area, effects, id_cols, peth=peth)
    if upset_sets:
        plot_upset(results, info, upset_sets, os.path.join(output_folder, 'upset'), scopes, id_cols=id_cols)
    print(f"[overlap] Done -> {output_folder}")


# ── preset: GLM (LRT) vs ROC (whisker_0) ─────────────────────────────────────

# (GLM reduced model, ROC analysis type) compared one to one; missing ones are skipped.
GLM_ROC_PAIRS = [
    (('GLM', 'whisker_encoding'), ('ROC', 'whisker_active')),
    (('GLM', 'whisker_first_bin'), ('ROC', 'whisker_active')),
    (('GLM', 'auditory_encoding'), ('ROC', 'auditory_active')),
    (('GLM', 'auditory_first_bin'), ('ROC', 'auditory_active')),
    (('GLM', 'licking_encoding'), ('ROC', 'spontaneous_licks')),
    (('GLM', 'lick_encoding'), ('ROC', 'spontaneous_licks')),
]
GLM_ROC_UPSET_SETS = {
    'GLM whisker': ('GLM', 'whisker_encoding'),
    'GLM whisker 50ms': ('GLM', 'whisker_first_bin'),
    'GLM auditory': ('GLM', 'auditory_encoding'),
    'GLM auditory 50ms': ('GLM', 'auditory_first_bin'),
    'GLM licking': ('GLM', 'licking_encoding'),
    'GLM reward': ('GLM', 'reward_encoding'),
    'ROC whisker_active': ('ROC', 'whisker_active'),
    'ROC auditory_active': ('ROC', 'auditory_active'),
    'ROC spontaneous_licks': ('ROC', 'spontaneous_licks'),
}


def run_glm_vs_roc(merged_df, output_folder, area_group_from_name=None, area_order=(), area_colors=None,
                   excluded_area_groups=(), pairs=GLM_ROC_PAIRS, upset_sets=GLM_ROC_UPSET_SETS,
                   roc_root=vq.ROC_ROOT, session_folder='whisker_0', peth=None, exclude_units=None):
    """GLM summaries of one filter set vs the ROC results of the same sessions."""
    info = neuron_info_from_glm(merged_df, area_group_from_name, excluded_area_groups)
    print(f"[overlap] Reading ROC results of {merged_df['mouse_id'].nunique()} mice ({session_folder}) ...")
    roc = roc_results(merged_df['mouse_id'].dropna().unique(), roc_root, session_folder)
    if roc.empty:
        print("[overlap] No ROC results found. Skipping.")
        return
    results = pd.concat([glm_results(merged_df), roc], ignore_index=True)
    run_overlap(results, info, output_folder, pairs=pairs, upset_sets=upset_sets,
                area_order=area_order, area_colors=area_colors, peth=peth, exclude_units=exclude_units)


# ── preset: reduced models of one GLM against each other ────────────────────

# (reduced model a, reduced model b) of the same fit; missing ones are skipped.
GLM_WITHIN_PAIRS = [
    ('whisker_encoding', 'whisker_first_bin'),
    ('auditory_encoding', 'auditory_first_bin'),
    ('whisker_encoding', 'auditory_encoding'),
    ('licking_encoding', 'jaw_onset_encoding'),
    ('licking_encoding', 'motor_encoding'),
    ('reward_encoding', 'licking_encoding'),
    ('reward_encoding', 'sum_rewards'),
    ('session_progress_encoding', 'sum_rewards'),
]
GLM_WITHIN_UPSET_SETS = {
    'whisker': 'whisker_encoding', 'auditory': 'auditory_encoding', 'licking': 'licking_encoding',
    'jaw onset': 'jaw_onset_encoding', 'reward': 'reward_encoding', 'session progress': 'session_progress_encoding',
}


def run_glm_within(merged_df, output_folder, area_group_from_name=None, area_order=(), area_colors=None,
                   excluded_area_groups=(), pairs=GLM_WITHIN_PAIRS, upset_sets=GLM_WITHIN_UPSET_SETS,
                   method='GLM', peth=None, exclude_units=None):
    """Reduced models of one GLM fit against each other: matrix of every
    model x every model, the pairs (model a, model b) and an UpSet of
    upset_sets ({label: model}). Effects: drop in test_corr / test_score of
    each model."""
    info = neuron_info_from_glm(merged_df, area_group_from_name, excluded_area_groups)
    results = glm_results(merged_df, method=method)
    run_overlap(results, info, output_folder, matrices=[(method, method)],
                pairs=[((method, a), (method, b)) for a, b in pairs],
                upset_sets={label: (method, model) for label, model in upset_sets.items()},
                area_order=area_order, area_colors=area_colors, peth=peth, exclude_units=exclude_units)


# ── preset: two GLM fits (e.g. two git versions) ─────────────────────────────

def run_glm_vs_glm(merged_a, merged_b, output_folder, label_a='GLM 1', label_b='GLM 2', pairs=None,
                   upset_models=('whisker_encoding', 'auditory_encoding', 'licking_encoding', 'reward_encoding'),
                   area_group_from_name=None, area_order=(), area_colors=None, excluded_area_groups=(),
                   peth=None, exclude_units=None):
    """Two GLM fits (e.g. two git versions, or two filter sets) against each
    other: matrix of every model of fit A x every model of fit B, pairs
    (model of A, model of B) - by default every model name the two fits share -
    and an UpSet of upset_models in both fits. Neurons: those fitted in both
    (each comparison uses the neurons with a result for both tests)."""
    if label_a == label_b:
        raise ValueError("label_a and label_b must differ (they name the two methods).")
    info = pd.concat([neuron_info_from_glm(m, area_group_from_name, excluded_area_groups)
                      for m in (merged_a, merged_b)], ignore_index=True)
    results = pd.concat([glm_results(merged_a, method=label_a), glm_results(merged_b, method=label_b)],
                        ignore_index=True)
    if pairs is None:
        tests = [set(results.loc[results['method'] == m, 'test']) for m in (label_a, label_b)]
        pairs = [(m, m) for m in sorted(tests[0] & tests[1])]
    upset = {f"{label} {model}": (label, model) for model in upset_models for label in (label_a, label_b)}
    run_overlap(results, info, output_folder, matrices=[(label_a, label_b)],
                pairs=[((label_a, a), (label_b, b)) for a, b in pairs], upset_sets=upset,
                area_order=area_order, area_colors=area_colors, peth=peth, exclude_units=exclude_units)


_PETH_CACHE = {}  # path -> load_peth_cache() output, so several runs read each cache once


def _peth_for_filter_set(filter_set_folder, mice, cache_file=PETH_CACHE_FILE):
    """The PETH cache of this filter set (same folder name in PETH_CACHE_ROOT), or None."""
    if not filter_set_folder:
        return None
    path = os.path.join(PETH_CACHE_ROOT, filter_set_folder, cache_file)
    if not os.path.exists(vq._long_path(path)):
        print(f"[overlap] No PETH cache {path}: no PSTH plots.")
        return None
    if path not in _PETH_CACHE:
        print(f"[overlap] Loading PETH cache {path} ...")
        _PETH_CACHE[path] = load_peth_cache(path)
    peth = _PETH_CACHE[path]
    return dict(table=peth['table'][peth['table']['mouse_id'].isin(set(mice))], time=peth['time'])


def main(mode, git_version, git_version_b=None, filter_set=None, mice=None, results_root=vq._MYRIAM_RESULTS,
         output=None, psth=True, peth_cache_file=PETH_CACHE_FILE, exclude_stim_artifact=True):
    """mode: 'glm_vs_roc' (GLM models vs ROC analysis types), 'glm_within'
    (the reduced models of one fit against each other) or 'glm_vs_glm'
    (git_version vs git_version_b). filter_set: 'good_units' / 'good_mua' /
    a filter-set folder name, or None for every filter set. mice: None for
    every mouse with summaries. output: None for
    {results_root}/neuron_set_overlap/{git}/{filter set}/{mode folder}
    ({git}_vs_{git b}/{filter set}/glm_version_overlap for glm_vs_glm).
    psth: also plot the PSTH of each set of each pair, from the PETH cache
    {PETH_CACHE_ROOT}/{filter-set folder}/{peth_cache_file} (default hit vs
    miss of psth_per_mouse_with_barplots.py).
    exclude_stim_artifact: leave out the neurons stim_artifact_units() flags in
    that PETH cache (isolated 0-5 ms jump on whisker hits and misses); the list
    is saved as excluded_units.txt in the output folder."""
    if mode not in ('glm_vs_roc', 'glm_within', 'glm_vs_glm'):
        raise ValueError(f"Unknown mode {mode!r}.")
    if mode == 'glm_vs_glm' and not git_version_b:
        raise ValueError("mode 'glm_vs_glm' needs git_version_b.")

    import glms_plotting_utils as gpu  # GLM summary loaders and area groups
    root = results_root
    versions = [git_version] + ([git_version_b] if mode == 'glm_vs_glm' else [])

    def _has_fit(mouse, version):
        return os.path.isdir(vq._long_path(os.path.join(root, mouse, 'whisker_0', 'unit_glm', version)))

    mice = mice or sorted(m for m in os.listdir(vq._long_path(root)) if all(_has_fit(m, v) for v in versions))
    aliases = {v: k for k, v in gpu.FILTER_SET_ALIASES.items()}
    filter_sets = gpu._discover_filter_sets(mice, root, git_version)
    if filter_set:
        fs = aliases.get(filter_set, filter_set)
        if fs not in filter_sets:
            raise ValueError(f"Unknown filter_set {filter_set!r}: use one of {sorted(aliases)} "
                             f"or a filter-set folder name of {git_version}.")
        filter_sets = [fs]
    area_kwargs = dict(area_group_from_name=gpu.allen.get_custom_area_groups_from_name(),
                       area_order=list(gpu.allen.get_custom_area_groups().keys()),
                       area_colors=gpu.allen.get_custom_area_groups_colors(),
                       excluded_area_groups=gpu.EXCLUDED_AREA_GROUPS)
    for fs in filter_sets:
        merged = [gpu._load_merged_summary_df(mice, root, v, fs) for v in versions]
        if any(m is None for m in merged):
            print(f"[overlap] No summaries for filter set {fs!r} in {versions}.")
            continue
        short = gpu._short_filter_set_name(fs) if fs else ''
        peth = _peth_for_filter_set(fs, mice, peth_cache_file) if (psth or exclude_stim_artifact) else None
        exclude = stim_artifact_units(peth) if (exclude_stim_artifact and peth is not None) else None
        if exclude_stim_artifact and peth is None:
            print("[overlap] WARNING: no PETH cache, so stim-artifact units are NOT excluded.")
        peth = peth if psth else None
        if mode == 'glm_vs_glm':
            out = output or os.path.join(root, 'neuron_set_overlap', f'{versions[0]}_vs_{versions[1]}',
                                         short, 'glm_version_overlap')
            run_glm_vs_glm(merged[0], merged[1], out, label_a=f'GLM {versions[0]}', label_b=f'GLM {versions[1]}',
                           peth=peth, exclude_units=exclude, **area_kwargs)
            continue
        folder = {'glm_vs_roc': 'glm_roc_overlap', 'glm_within': 'glm_model_overlap'}[mode]
        out = output or os.path.join(root, 'neuron_set_overlap', git_version, short, folder)
        (run_glm_vs_roc if mode == 'glm_vs_roc' else run_glm_within)(merged[0], out, peth=peth,
                                                                     exclude_units=exclude, **area_kwargs)


if __name__ == '__main__':
    # Edit here. Each entry is one run (see main for the options).
    runs = [
        dict(mode='glm_vs_roc', git_version='d210c27', filter_set='good_mua'),
        dict(mode='glm_within', git_version='d210c27', filter_set='good_mua'),
        dict(mode='glm_vs_glm', git_version='d210c27', git_version_b='e78c1f4', filter_set='good_mua'),
    ]
    mice = None  # None = every mouse with summaries, or e.g. ['AB161', 'AB085']
    for run in runs:
        main(mice=mice, **run)
