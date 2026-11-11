"""
visualize_quantity.py - show a per-neuron quantity on coronal slices of the Allen atlas.

Every function takes a DataFrame with one row per neuron and its Allen CCF
coordinates in um (ccf_atlas_ap / ccf_atlas_ml / ccf_atlas_dv), used as stored:
they are in the space of the brainglobe annotation volume drawn underneath.

Neurons are assigned to coronal slabs (slab_um thick, centred on ap_positions)
and each slab is projected onto the atlas section at its centre. Rows of a
figure are the reward groups (R+, R-) when a reward-group column is present;
a pooled version (all neurons together, orange) is always made.

Public functions
----------------
plot_group_density      where the neurons of a group are: density, points, and
                        density over the sampled space (light grey).
plot_fraction_on_slices percentage of neurons with a True flag (e.g. % selective),
                        locally, where enough neurons were recorded.
plot_value_on_slices    one dot per neuron, coloured by a signed value.
plot_area_group_values_on_slices
                        every area group painted with one value (with
                        area_group_means_over_mice: the mean over mice).
plot_roc_selectivity    fraction, density and area-group maps for each
                        analysis_type of a ROC results table.

Needs numpy, pandas, matplotlib, scipy and brainglobe_atlasapi (first use
downloads the atlas). Without brainglobe_atlasapi the plots are skipped with a
message. Figures are saved as PNG and PDF.
"""

import functools
import itertools
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CCF_COLS = ['ccf_atlas_ap', 'ccf_atlas_ml', 'ccf_atlas_dv']
REWARD_GROUP_COLORS = {1: '#00b400', 0: '#c800c8'}  # reward_group: 1 = R+, 0 = R-
POOLED_COLOR = '#F0A50F'

# Colors for learners vs experts comparison barplots
LEARNING_COLORS = {
    ('learners', 'R+'): '#45624F',
    ('learners', 'R-'): '#861886',
    ('experts',  'R+'): '#00B400',
    ('experts',  'R-'): '#C903CA',
}
# A figure row is one value of the row column (reward_col): 1 / 0 for the reward
# groups, or 'learner R+', 'expert R+', 'learner R-', 'expert R-'. Rows are
# drawn in ROW_ORDER; 'pooled' is every neuron that belongs to one of them.
ROW_ORDER = [1, 0] + [f'{learning} {reward}' for reward in ('R+', 'R-') for learning in ('learner', 'expert')]
# LEARNING_COLORS may be keyed 'learner' / 'expert' or 'learners' / 'experts'; rows use the singular.
ROW_COLORS = {**REWARD_GROUP_COLORS, 'pooled': POOLED_COLOR,
              **{f"{learning.rstrip('s')} {reward}": color for (learning, reward), color in LEARNING_COLORS.items()}}
assert all(row in ROW_COLORS for row in ROW_ORDER), "every figure row needs a colour in ROW_COLORS"
ROW_LABELS = {1: 'R+', 0: 'R-', 'pooled': 'pooled', **{row: row for row in ROW_ORDER[2:]}}
SAMPLED_SPACE_COLOR = '#dedede'
DEFAULT_AP_POSITIONS = (3000, 4000, 5000, 6000, 7000, 8000, 9000)
DEFAULT_ATLAS = 'allen_mouse_25um'


# ── atlas ────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=4)
def atlas_coronal_slices(ap_positions, atlas_name=DEFAULT_ATLAS):
    """Coronal annotation slices of the brainglobe atlas at ap_positions (a
    tuple, CCF um).

    Returns dict(res_dv, res_ml, n_dv, n_ml, width_um, height_um, midline_um,
    slices={pos: dict(labels, brain, edges)}), or None (with a message) if
    brainglobe_atlasapi isn't installed. Cached, so the atlas volume is read
    once per process.
    """
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
    except ImportError:
        print("[brain_slices] brainglobe_atlasapi not installed in this env "
              "(pip install brainglobe-atlasapi). Skipping brain-slice plots.")
        return None
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    res_ap, res_dv, res_ml = atlas.resolution
    annotation = atlas.annotation  # (AP, DV, ML) voxels
    n_ap, n_dv, n_ml = annotation.shape
    slices = {}
    for pos in ap_positions:
        ap_idx = int(round(pos / res_ap))
        if not 0 <= ap_idx < n_ap:
            continue
        sl = np.array(annotation[ap_idx])
        edges = np.zeros(sl.shape, dtype=bool)
        edges[:-1, :] |= sl[:-1, :] != sl[1:, :]
        edges[:, :-1] |= sl[:, :-1] != sl[:, 1:]
        slices[pos] = dict(labels=sl, brain=sl > 0,
                           edges=np.ma.masked_where(~edges, edges.astype(float)))
    return dict(res_dv=res_dv, res_ml=res_ml, n_dv=n_dv, n_ml=n_ml,
                width_um=n_ml * res_ml, height_um=n_dv * res_dv, midline_um=n_ml * res_ml / 2,
                slices=slices)


def draw_atlas_outline(ax, atlas, pos, x_max):
    """Region borders (light grey) and brain outline (black) of one coronal
    slice, in CCF um with dorsal up; crops the panel to ML 0..x_max."""
    extent = [0, atlas['width_um'], atlas['height_um'], 0]
    ax.imshow(atlas['slices'][pos]['edges'], extent=extent, cmap='Greys', vmin=0, vmax=3,
              interpolation='nearest', zorder=2)
    ax.contour(np.arange(atlas['n_ml']) * atlas['res_ml'] + atlas['res_ml'] / 2,
               np.arange(atlas['n_dv']) * atlas['res_dv'] + atlas['res_dv'] / 2,
               atlas['slices'][pos]['brain'], levels=[0.5], colors='black',
               linewidths=0.5, zorder=3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(atlas['height_um'], 0)
    ax.set_aspect('equal')
    ax.axis('off')


# ── shared helpers ───────────────────────────────────────────────────────────

def _long_path(path):
    """Extended-length form of a Windows path, so nested output folders can go
    past the 260-character limit. Unchanged on other systems."""
    if os.name != 'nt':
        return path
    path = os.path.abspath(path)
    if path.startswith('\\\\?\\'):
        return path
    if path.startswith('\\\\'):
        return '\\\\?\\UNC\\' + path[2:]
    return '\\\\?\\' + path


def save_figure(fig, output_dir, filename, file_formats=('png', 'pdf')):
    output_dir = _long_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for file_format in file_formats:
        fig.savefig(os.path.join(output_dir, f"{filename}.{file_format}"), bbox_inches='tight', dpi=300)


def _save_table(df, output_dir, filename):
    output_dir = _long_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), sep='\t', index=False)


def _one_row_per_neuron(neurons, id_cols, reward_col):
    """Rows with coordinates, one per neuron; reward groups as int where known."""
    keep = [c for c in neurons.columns]
    df = neurons[keep].dropna(subset=CCF_COLS).drop_duplicates(subset=list(id_cols)).copy()
    df[CCF_COLS] = df[CCF_COLS].astype(float)
    if reward_col in df.columns:
        df[reward_col] = df[reward_col].replace({'R+': 1, 'R-': 0})
    return df


def _fold(df, atlas, fold_hemispheres):
    """ML mirrored onto one hemisphere (so mice recorded on either side pool);
    returns (df, right-hand x limit of the panels)."""
    if not fold_hemispheres:
        return df, atlas['width_um']
    ml = df['ccf_atlas_ml'].to_numpy(dtype=float)
    df = df.assign(ccf_atlas_ml=np.where(ml > atlas['midline_um'], 2 * atlas['midline_um'] - ml, ml))
    return df, atlas['midline_um'] + 300


def _row_sets(df, reward_col):
    """Figure row sets: one row per reward group present (if the column
    exists), and always a single pooled row (file suffix '_pooled')."""
    sets = []
    if reward_col in df.columns:
        present = [row for row in ROW_ORDER if (df[reward_col] == row).any()]
        if present:
            sets.append((present, ''))
    sets.append((['pooled'], '_pooled'))
    return sets


def _row_df(df, row, reward_col):
    if row == 'pooled':
        return df[df[reward_col].isin(ROW_ORDER)] if reward_col in df.columns else df
    return df[df[reward_col] == row]


def _row_color(row):
    return ROW_COLORS.get(row, '#444444')


class _Grid:
    """(DV, ML) grid of bin_um bins over an atlas slice, with the brain mask
    of each slice sampled on it."""

    def __init__(self, atlas, bin_um):
        self.bin_um = bin_um
        self.ml_edges = np.arange(0, atlas['width_um'] + bin_um, bin_um)
        self.dv_edges = np.arange(0, atlas['height_um'] + bin_um, bin_um)
        ml_c = (self.ml_edges[:-1] + self.ml_edges[1:]) / 2
        dv_c = (self.dv_edges[:-1] + self.dv_edges[1:]) / 2
        ml_vox = np.clip((ml_c / atlas['res_ml']).astype(int), 0, atlas['n_ml'] - 1)
        dv_vox = np.clip((dv_c / atlas['res_dv']).astype(int), 0, atlas['n_dv'] - 1)
        self.brain = {pos: info['labels'][np.ix_(dv_vox, ml_vox)] > 0 for pos, info in atlas['slices'].items()}
        self.extent = [0, atlas['width_um'], atlas['height_um'], 0]
        self.bin_area_mm2 = (bin_um / 1000) ** 2

    def counts(self, df, weights=None):
        """Neurons per bin, or the sum of `weights` (one per row of df) per bin."""
        hist, _, _ = np.histogram2d(df['ccf_atlas_dv'], df['ccf_atlas_ml'], bins=[self.dv_edges, self.ml_edges],
                                    weights=weights)
        return hist

    def sampled_space(self, df, pos, radius_um):
        """Masked array, 1 where a bin is within radius_um of a neuron of df."""
        from scipy.ndimage import binary_dilation
        radius = max(1, int(round(radius_um / self.bin_um)))
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = binary_dilation(self.counts(df) > 0, structure=(xx ** 2 + yy ** 2) <= radius ** 2) & self.brain[pos]
        return np.ma.masked_where(~mask, np.ones(mask.shape))


def _in_slab(df, pos, slab_um):
    return df[np.abs(df['ccf_atlas_ap'] - pos) < slab_um / 2]


def _colorbar(fig, row_axes, cmap, vmin, vmax, label, extend='neither', orientation='vertical'):
    mappable = matplotlib.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
    cbar = fig.colorbar(mappable, ax=row_axes, shrink=0.8, pad=0.01 if orientation == 'vertical' else 0.03,
                        extend=extend, orientation=orientation)
    cbar.set_label(label, fontsize=6)
    cbar.ax.tick_params(labelsize=5)


def _new_figure(n_rows, positions, x_max, atlas):
    panel_w = 2.4 * x_max / atlas['width_um']
    return plt.subplots(n_rows, len(positions), figsize=(panel_w * len(positions) + 1.5, 2.0 * n_rows + 0.6),
                        squeeze=False, gridspec_kw=dict(wspace=0.03, hspace=0.1))


# ── where a group of neurons is ──────────────────────────────────────────────

def plot_group_density(
        neurons, groups, output_folder,
        ap_positions=DEFAULT_AP_POSITIONS, slab_um=1000, bin_um=50, sigma_um=100,
        fold_hemispheres=True, atlas_name=DEFAULT_ATLAS, sampled_radius_um=100,
        reward_col='reward_group', id_cols=('mouse_id', 'neuron_id'),
        file_prefix='selectivity', all_name='all_neurons'):
    """Spatial distribution of groups of neurons on coronal atlas slices.

    neurons : one row per recorded neuron (id_cols, CCF_COLS, optionally
        reward_col). This is the sampling: every neuron that could have been
        in a group.
    groups : {name: DataFrame with id_cols} - the neurons of each group (a
        subset of `neurons`). A group named all_name with every neuron is
        added in front.

    Three versions per group, each in its own subfolder of output_folder:
      density/                smoothed density of the group
      points/                 one dot per neuron of the group
      density_with_sampling/  the density over the sampled space (flat light
                              grey: within sampled_radius_um of any recorded
                              neuron); not made for the all_name group
    each as one row per reward group, and as a pooled row (file suffix
    '_pooled', orange) with its own colour scale. Every figure is also saved in
    a vertical layout (file suffix '_vertical'): one column per row group, the
    slices stacked top to bottom, colour bars below each column. When the rows are learner /
    expert of each reward group, the two density versions are also saved with
    file suffix '_expert_scale': both rows of a reward group in the expert's
    colour and on one scale per reward group, with one colour bar for the pair
    (the default version has one colour per row and one scale for all rows).

    Density = 2D histogram of (ML, DV) in bin_um bins, Gaussian-smoothed with
    sigma_um, divided by the group's neuron count in that row (all slabs) and
    by the bin area: fraction of the group's neurons per mm² of the coronal
    plane. Neurons are pooled across mice. A neuron belongs to every slab whose
    centre is within slab_um / 2 of its AP position.

    Also writes neuron_counts_per_slab.txt. Returns None; skipped with a
    message if coordinates or brainglobe_atlasapi are missing.
    """
    from scipy.ndimage import gaussian_filter

    missing = [c for c in CCF_COLS if c not in neurons.columns]
    if missing:
        print(f"[plot_group_density] Input lacks {missing}. Skipping.")
        return
    atlas = atlas_coronal_slices(tuple(ap_positions), atlas_name)
    if atlas is None or not atlas['slices']:
        return
    positions = list(atlas['slices'])
    id_cols = list(id_cols)

    neurons = _one_row_per_neuron(neurons, id_cols, reward_col)
    neurons, x_max = _fold(neurons, atlas, fold_hemispheres)
    grid = _Grid(atlas, bin_um)
    sigma_bins = sigma_um / bin_um

    all_groups = [(all_name, neurons)]
    for name, members in groups.items():
        member_df = neurons.merge(members[id_cols].drop_duplicates(), on=id_cols)
        if not member_df.empty:
            all_groups.append((name, member_df))

    def _densities(df, rows):
        out = {}
        for row in rows:
            row_df = _row_df(df, row, reward_col)
            for pos in positions:
                in_slab = _in_slab(row_df, pos, slab_um)
                dens = gaussian_filter(grid.counts(in_slab), sigma_bins) / max(len(row_df), 1) / grid.bin_area_mm2
                dens[~grid.brain[pos]] = np.nan
                out[(row, pos)] = (in_slab, dens)
        values = np.concatenate([d[np.isfinite(d)] for _, d in out.values()])
        vmax = np.percentile(values[values > 0], 99.5) if (values > 0).any() else 1.0
        return out, vmax

    row_sets = _row_sets(neurons, reward_col)
    sampled = {(row, pos): grid.sampled_space(_in_slab(_row_df(neurons, row, reward_col), pos, slab_um),
                                             pos, sampled_radius_um)
               for rows, _ in row_sets for row in rows for pos in positions}
    sampled_cmap = mcolors.ListedColormap([SAMPLED_SPACE_COLOR])
    count_rows = []

    for (group_name, group_df), (rows, suffix) in itertools.product(all_groups, row_sets):
        slabs, vmax = _densities(group_df, rows)
        for row in rows:
            for pos in positions:
                count_rows.append(dict(group=group_name, reward_group=ROW_LABELS[row], ap_position_um=pos,
                                       n_neurons_slab=len(slabs[(row, pos)][0]),
                                       n_neurons_group=len(_row_df(group_df, row, reward_col))))

        # Second colour scheme when rows are learner / expert of a reward group: both rows
        # of a reward group in the expert's colour, on one scale per reward group (the
        # 99.5th percentile over its two rows), with a single colour bar for the pair.
        reward_of = {row: row.split(' ', 1)[1] for row in rows
                     if isinstance(row, str) and row.split(' ', 1)[0] in ('learner', 'expert')}
        pair_vmax = {}
        for reward in set(reward_of.values()):
            values = np.concatenate([slabs[(row, pos)][1][np.isfinite(slabs[(row, pos)][1])]
                                     for row in rows if reward_of.get(row) == reward for pos in positions])
            pair_vmax[reward] = np.percentile(values[values > 0], 99.5) if (values > 0).any() else 1.0
        schemes = [False] + ([True] if reward_of else [])

        versions = ['density', 'points'] + ([] if group_name == all_name else ['density_with_sampling'])
        # 'horizontal': one row per group of neurons, slices left to right (default files).
        # 'vertical': one column per group, slices top to bottom (file suffix '_vertical').
        for version, expert_scale, layout in itertools.product(versions, schemes, ('horizontal', 'vertical')):
            if expert_scale and version == 'points':
                continue
            vertical = layout == 'vertical'
            if vertical:
                panel_w = 2.4 * x_max / atlas['width_um']
                fig, axes_t = plt.subplots(len(positions), len(rows), squeeze=False,
                                           figsize=(panel_w * len(rows) + 1.0, 2.0 * len(positions) + 1.2),
                                           gridspec_kw=dict(wspace=0.05, hspace=0.12))
                axes = axes_t.T  # axes[r, c]: group r, slice c, as in the horizontal layout
            else:
                fig, axes = _new_figure(len(rows), positions, x_max, atlas)
            for r, row in enumerate(rows):
                shared = expert_scale and row in reward_of
                color = _row_color(f'expert {reward_of[row]}') if shared else _row_color(row)
                row_vmax = pair_vmax[reward_of[row]] if shared else vmax
                rgb = mcolors.to_rgb(color)
                if version == 'density_with_sampling':
                    # Transparent -> colour, so the grey sampled space shows through.
                    cmap = mcolors.LinearSegmentedColormap.from_list(f'dens_{row}', [(*rgb, 0.0), (*rgb, 1.0)])
                else:
                    cmap = mcolors.LinearSegmentedColormap.from_list(f'dens_{row}', ['#ffffff', color])
                cmap.set_bad(alpha=0)
                for c, pos in enumerate(positions):
                    ax = axes[r, c]
                    in_slab, dens = slabs[(row, pos)]
                    if version == 'points':
                        ax.scatter(in_slab['ccf_atlas_ml'], in_slab['ccf_atlas_dv'], s=2, color=color,
                                   alpha=0.6, linewidths=0, zorder=1, rasterized=True)
                    else:
                        if version == 'density_with_sampling':
                            ax.imshow(sampled[(row, pos)], extent=grid.extent, cmap=sampled_cmap,
                                      interpolation='nearest', zorder=0.5)
                        ax.imshow(dens, extent=grid.extent, cmap=cmap, vmin=0, vmax=row_vmax,
                                  interpolation='bilinear', zorder=1)
                    draw_atlas_outline(ax, atlas, pos, x_max)
                    n_slab = len(in_slab)
                    if vertical:
                        ax.set_title(f"n={n_slab}", fontsize=6)
                        if r == 0:
                            ax.text(-0.08, 0.5, f"AP {pos} µm", transform=ax.transAxes, ha='right',
                                    va='center', fontsize=7)
                    else:
                        ax.set_title(f"AP {pos} µm\nn={n_slab}" if r == 0 else f"n={n_slab}", fontsize=6)
                group_label = f"{ROW_LABELS[row]}\n(n={len(_row_df(group_df, row, reward_col))})"
                if vertical:
                    axes[r, 0].text(0.5, 1.25, group_label, transform=axes[r, 0].transAxes, ha='center',
                                    va='bottom', fontsize=8)
                else:
                    axes[r, 0].text(-0.08, 0.5, group_label, transform=axes[r, 0].transAxes, ha='right',
                                    va='center', fontsize=8)
                if version == 'points':
                    continue
                orientation = 'horizontal' if vertical else 'vertical'
                if shared:
                    # One colour bar for the reward group's rows, drawn with its last row.
                    pair = [i for i, other in enumerate(rows) if reward_of.get(other) == reward_of[row]]
                    if r == pair[-1]:
                        _colorbar(fig, axes[pair, :].ravel().tolist(), cmap, 0, row_vmax, 'fraction of group / mm²',
                                  orientation=orientation)
                else:
                    _colorbar(fig, axes[r, :].tolist(), cmap, 0, row_vmax, 'fraction of group / mm²',
                              orientation=orientation)

            title = f"{version.replace('_', ' ')} - {group_name} (slab {slab_um} µm)"
            if expert_scale:
                title += "; learner and expert on the same scale, per reward group"
            if version == 'density_with_sampling':
                title += f"\nlight grey = sampled space (within {sampled_radius_um} µm of a recorded neuron)"
            fig.suptitle(title, fontsize=10)
            save_figure(fig, os.path.join(output_folder, version),
                        f"{file_prefix}_{version}_{group_name}{suffix}{'_expert_scale' if expert_scale else ''}"
                        f"{'_vertical' if vertical else ''}")
            plt.close(fig)

    _save_table(pd.DataFrame(count_rows), output_folder, 'neuron_counts_per_slab.txt')


# ── percentage of neurons with a flag ────────────────────────────────────────

def plot_fraction_on_slices(
        neurons, flag_col, output_folder, name=None, label=None,
        ap_positions=DEFAULT_AP_POSITIONS, slab_um=1000, bin_um=50, sigma_um=150, min_neurons=5,
        vmax=None, fold_hemispheres=True, atlas_name=DEFAULT_ATLAS, sampled_radius_um=100,
        reward_col='reward_group', id_cols=('mouse_id', 'neuron_id')):
    """Local percentage of neurons whose flag_col is True (e.g. % selective).

    neurons : one row per recorded neuron (id_cols, CCF_COLS, a boolean
        flag_col, optionally reward_col).

    In each slab, the flagged neurons and all neurons are each counted in
    bin_um bins and smoothed with the same sigma_um Gaussian; the map is
    100 * smoothed flagged / smoothed all. It is only drawn where the smoothed
    count of all neurons corresponds to at least min_neurons neurons under the
    Gaussian (count * 2 * pi * sigma², in bins); elsewhere the sampled space is
    left light grey (recorded, but too few neurons for a percentage) or white
    (not recorded). Unlike a density, this is not driven by where the probes
    were. Neurons are pooled across mice.

    One figure with a row per reward group and one pooled (suffix '_pooled'),
    sharing a colour scale 0..vmax (default: the 99th percentile of the drawn
    values over both figures, at most 100). Each panel title gives the slab's
    neuron count and its overall percentage; a table of those numbers is
    written next to the figures.

    Files: {output_folder}/fraction_{name}[_pooled].png/.pdf and
    fraction_{name}_per_slab.txt. name defaults to flag_col.
    """
    from scipy.ndimage import gaussian_filter

    missing = [c for c in CCF_COLS + [flag_col] if c not in neurons.columns]
    if missing:
        print(f"[plot_fraction_on_slices] Input lacks {missing}. Skipping.")
        return
    atlas = atlas_coronal_slices(tuple(ap_positions), atlas_name)
    if atlas is None or not atlas['slices']:
        return
    positions = list(atlas['slices'])
    name = name or flag_col
    label = label or f"% of neurons with {flag_col}"

    neurons = _one_row_per_neuron(neurons, list(id_cols), reward_col)
    neurons = neurons.assign(_flag=neurons[flag_col].fillna(False).astype(bool))
    neurons, x_max = _fold(neurons, atlas, fold_hemispheres)
    grid = _Grid(atlas, bin_um)
    sigma_bins = sigma_um / bin_um
    kernel_bins = 2 * np.pi * sigma_bins ** 2  # neurons "under" the Gaussian = smoothed count * this
    row_sets = _row_sets(neurons, reward_col)

    maps, table = {}, []
    for rows, _ in row_sets:
        for row in rows:
            row_df = _row_df(neurons, row, reward_col)
            for pos in positions:
                in_slab = _in_slab(row_df, pos, slab_um)
                all_sm = gaussian_filter(grid.counts(in_slab), sigma_bins)
                flag_sm = gaussian_filter(grid.counts(in_slab[in_slab['_flag']]), sigma_bins)
                enough = (all_sm * kernel_bins >= min_neurons) & grid.brain[pos]
                pct = np.full(all_sm.shape, np.nan)
                pct[enough] = 100 * flag_sm[enough] / all_sm[enough]
                n, n_flag = len(in_slab), int(in_slab['_flag'].sum())
                maps[(row, pos)] = (pct, grid.sampled_space(in_slab, pos, sampled_radius_um), n, n_flag)
                table.append(dict(flag=name, reward_group=ROW_LABELS[row], ap_position_um=pos, n_neurons=n,
                                  n_flagged=n_flag, percent=100 * n_flag / n if n else np.nan))

    drawn = np.concatenate([m[0][np.isfinite(m[0])] for m in maps.values()])
    if vmax is None:
        vmax = min(100.0, float(np.percentile(drawn, 99))) if len(drawn) else 100.0
        vmax = max(vmax, 1e-6)
    sampled_cmap = mcolors.ListedColormap([SAMPLED_SPACE_COLOR])

    for rows, suffix in row_sets:
        fig, axes = _new_figure(len(rows), positions, x_max, atlas)
        for r, row in enumerate(rows):
            color = _row_color(row)
            # Start at a light tint, not white, so "drawn, about 0 %" differs from "not drawn".
            tint = tuple(1 - 0.12 * (1 - c) for c in mcolors.to_rgb(color))
            cmap = mcolors.LinearSegmentedColormap.from_list(f'frac_{row}', [tint, color])
            cmap.set_bad(alpha=0)
            row_df = _row_df(neurons, row, reward_col)
            for c, pos in enumerate(positions):
                ax = axes[r, c]
                pct, sampled, n, n_flag = maps[(row, pos)]
                ax.imshow(sampled, extent=grid.extent, cmap=sampled_cmap, interpolation='nearest', zorder=0.5)
                ax.imshow(pct, extent=grid.extent, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest', zorder=1)
                draw_atlas_outline(ax, atlas, pos, x_max)
                slab_pct = f"{100 * n_flag / n:.0f}%" if n else "-"
                ax.set_title((f"AP {pos} µm\n" if r == 0 else "") + f"n={n}, {slab_pct}", fontsize=6)
            n_row, n_row_flag = len(row_df), int(row_df['_flag'].sum())
            row_pct = f"{100 * n_row_flag / n_row:.1f}%" if n_row else "-"
            axes[r, 0].text(-0.08, 0.5, f"{ROW_LABELS[row]}\n{n_row_flag}/{n_row}\n({row_pct})",
                            transform=axes[r, 0].transAxes, ha='right', va='center', fontsize=8)
            clipped = bool(len(drawn) and drawn.max() > vmax)
            _colorbar(fig, axes[r, :].tolist(), cmap, 0, vmax, label, extend='max' if clipped else 'neither')
        fig.suptitle(f"{label} - {name} (slab {slab_um} µm, smoothing {sigma_um} µm)\n"
                     f"drawn where >= {min_neurons} neurons under the smoothing kernel; "
                     f"light grey = sampled, too few neurons", fontsize=9)
        save_figure(fig, output_folder, f'fraction_{name}{suffix}')
        plt.close(fig)

    _save_table(pd.DataFrame(table), output_folder, f'fraction_{name}_per_slab.txt')


# ── a value per neuron: dots, and its local mean ─────────────────────────────

def plot_value_maps_on_slices(
        neurons, value_col, output_folder, name=None, label=None, diverging=True,
        ap_positions=DEFAULT_AP_POSITIONS, slab_um=1000, bin_um=50, sigma_um=150, min_neurons=5,
        fold_hemispheres=True, atlas_name=DEFAULT_ATLAS, sampled_radius_um=100,
        reward_col='reward_group', id_cols=('mouse_id', 'neuron_id')):
    """A per-neuron value (e.g. ROC selectivity) on coronal atlas slices.

    neurons : one row per recorded neuron (id_cols, CCF_COLS, value_col,
        optionally reward_col). Rows without a value are left out.

    Two versions, each in its own subfolder of output_folder:
      points/  one dot per neuron at its (ML, DV), coloured by its value
      map/     local mean of the value: in each slab, the sum of the values and
               the neuron count are each binned (bin_um) and smoothed with the
               same sigma_um Gaussian, and the map is smoothed sum / smoothed
               count. Like plot_fraction_on_slices it is only drawn where at
               least min_neurons neurons lie under the Gaussian; elsewhere the
               sampled space is light grey. Not driven by where the probes were.
    each as one row per reward group, and as a pooled row (suffix '_pooled').
    Neurons are pooled across mice.

    diverging : True for a signed value (blue-red scale centred on 0); False
        for a value >= 0 (light to the row's colour, from 0).
    The scale is shared by every row of both figures of a version: the 98th
    percentile of |value| for the points, the 99th percentile of the drawn
    |local mean| for the map (a local mean is smaller than single values).

    Files: {output_folder}/points/{name}_points[_pooled].png/.pdf,
    {output_folder}/map/{name}_map[_pooled].png/.pdf and {name}_per_slab.txt.
    name defaults to value_col.
    """
    from scipy.ndimage import gaussian_filter

    missing = [c for c in CCF_COLS + [value_col] if c not in neurons.columns]
    if missing:
        print(f"[plot_value_maps_on_slices] Input lacks {missing}. Skipping.")
        return
    atlas = atlas_coronal_slices(tuple(ap_positions), atlas_name)
    if atlas is None or not atlas['slices']:
        return
    positions = list(atlas['slices'])
    name = name or value_col
    label = label or value_col

    neurons = _one_row_per_neuron(neurons, list(id_cols), reward_col)
    neurons = neurons.assign(_value=pd.to_numeric(neurons[value_col], errors='coerce')).dropna(subset=['_value'])
    if neurons.empty:
        print(f"[plot_value_maps_on_slices] No neuron has a {value_col}. Skipping.")
        return
    neurons, x_max = _fold(neurons, atlas, fold_hemispheres)
    grid = _Grid(atlas, bin_um)
    sigma_bins = sigma_um / bin_um
    kernel_bins = 2 * np.pi * sigma_bins ** 2  # neurons "under" the Gaussian = smoothed count * this
    row_sets = _row_sets(neurons, reward_col)

    slabs, table = {}, []
    for rows, _ in row_sets:
        for row in rows:
            row_df = _row_df(neurons, row, reward_col)
            for pos in positions:
                in_slab = _in_slab(row_df, pos, slab_um)
                all_sm = gaussian_filter(grid.counts(in_slab), sigma_bins)
                sum_sm = gaussian_filter(grid.counts(in_slab, weights=in_slab['_value']), sigma_bins)
                enough = (all_sm * kernel_bins >= min_neurons) & grid.brain[pos]
                local_mean = np.full(all_sm.shape, np.nan)
                local_mean[enough] = sum_sm[enough] / all_sm[enough]
                slabs[(row, pos)] = (in_slab, local_mean, grid.sampled_space(in_slab, pos, sampled_radius_um))
                table.append(dict(value=name, reward_group=ROW_LABELS[row], ap_position_um=pos,
                                  n_neurons=len(in_slab), mean=in_slab['_value'].mean(),
                                  median=in_slab['_value'].median()))

    drawn = np.abs(np.concatenate([m[np.isfinite(m)] for _, m, _ in slabs.values()]))
    vlims = {'points': max(float(np.percentile(np.abs(neurons['_value']), 98)), 1e-9),
             'map': max(float(np.percentile(drawn, 99)), 1e-9) if len(drawn) else 1.0}
    sampled_cmap = mcolors.ListedColormap([SAMPLED_SPACE_COLOR])

    for version, (rows, suffix) in itertools.product(('points', 'map'), row_sets):
        vlim = vlims[version]
        vmin = -vlim if diverging else 0.0
        fig, axes = _new_figure(len(rows), positions, x_max, atlas)
        for r, row in enumerate(rows):
            if diverging:
                cmap = plt.get_cmap('coolwarm').copy()  # grey midpoint: near-zero stays visible on white
            else:
                color = _row_color(row)
                tint = tuple(1 - 0.2 * (1 - c) for c in mcolors.to_rgb(color))
                cmap = mcolors.LinearSegmentedColormap.from_list(f'value_{row}', [tint, color])
            cmap.set_bad(alpha=0)
            row_df = _row_df(neurons, row, reward_col)
            for c, pos in enumerate(positions):
                ax = axes[r, c]
                in_slab, local_mean, sampled = slabs[(row, pos)]
                if version == 'points':
                    # Random draw order so no value systematically hides the others.
                    shuffled = in_slab.sample(frac=1, random_state=0)
                    ax.scatter(shuffled['ccf_atlas_ml'], shuffled['ccf_atlas_dv'], c=shuffled['_value'],
                               cmap=cmap, vmin=vmin, vmax=vlim, s=3, alpha=0.8, linewidths=0, zorder=1,
                               rasterized=True)
                else:
                    ax.imshow(sampled, extent=grid.extent, cmap=sampled_cmap, interpolation='nearest', zorder=0.5)
                    ax.imshow(local_mean, extent=grid.extent, cmap=cmap, vmin=vmin, vmax=vlim,
                              interpolation='nearest', zorder=1)
                draw_atlas_outline(ax, atlas, pos, x_max)
                slab_mean = f"{in_slab['_value'].mean():.2f}" if len(in_slab) else "-"
                ax.set_title((f"AP {pos} µm\n" if r == 0 else "") + f"n={len(in_slab)}, mean {slab_mean}",
                             fontsize=6)
            axes[r, 0].text(-0.08, 0.5, f"{ROW_LABELS[row]}\n(n={len(row_df)})",
                            transform=axes[r, 0].transAxes, ha='right', va='center', fontsize=8)
            values = np.abs(row_df['_value']) if version == 'points' else drawn
            clipped = bool(len(values) and values.max() > vlim)
            extend = ('both' if diverging else 'max') if clipped else 'neither'
            _colorbar(fig, axes[r, :].tolist(), cmap, vmin, vlim, label, extend=extend)
        if version == 'points':
            title = f"{label}: one dot per neuron (slab {slab_um} µm)"
        else:
            title = (f"{label}: local mean (slab {slab_um} µm, smoothing {sigma_um} µm)\n"
                     f"drawn where >= {min_neurons} neurons under the smoothing kernel; "
                     f"light grey = sampled, too few neurons")
        fig.suptitle(title, fontsize=9)
        save_figure(fig, os.path.join(output_folder, version), f'{name}_{version}{suffix}')
        plt.close(fig)

    _save_table(pd.DataFrame(table), output_folder, f'{name}_per_slab.txt')


# ── one dot per neuron, coloured by a value ──────────────────────────────────

def plot_value_on_slices(
        neurons, value_col, value_label, output_folder, title, row_labels=None,
        ap_positions=DEFAULT_AP_POSITIONS, slab_um=1000, bound=None,
        fold_hemispheres=True, atlas_name=DEFAULT_ATLAS, reward_col='reward_group',
        file_prefix='brain_slices'):
    """One figure per reward group: coronal slabs, one dot per neuron at its
    (ML, DV), coloured by value_col on a diverging scale centred on 0 and
    shared by both reward groups.

    bound : largest absolute value the quantity can take (e.g. 100 for a %,
        1 for a modulation index). If given, the scale is +/-bound, with a
        second version at +/-bound/2 (file suffix _pm{bound/2}). If None, the
        scale is +/- the 98th percentile of |value|, with a second version
        clipped at the 90th percentile (suffix _clipped). Values beyond the
        scale take the end colours.

    Skipped if neurons has no coordinates or brainglobe_atlasapi isn't installed.
    """
    row_labels = row_labels or ROW_LABELS
    if any(c not in neurons.columns for c in CCF_COLS) or value_col not in neurons.columns:
        return
    df = neurons.dropna(subset=CCF_COLS + [value_col])
    if df.empty:
        return
    atlas = atlas_coronal_slices(tuple(ap_positions), atlas_name)
    if atlas is None or not atlas['slices']:
        return
    positions = list(atlas['slices'])
    df, x_max = _fold(df, atlas, fold_hemispheres)

    if bound is not None:
        scales = [(bound, ''), (bound / 2, f'_pm{bound / 2:g}')]
    else:
        abs_vals = np.abs(df[value_col])
        scales = [(max(np.percentile(abs_vals, 98), 1e-9), ''),
                  (max(np.percentile(abs_vals, 90), 1e-9), '_clipped')]
    cmap = 'coolwarm'  # grey midpoint, so near-zero neurons stay visible on white
    panel_w = 2.4 * x_max / atlas['width_um']

    for (vlim, suffix), rg in itertools.product(scales, [1, 0]):
        norm = mcolors.Normalize(-vlim, vlim)
        rg_df = df[df[reward_col] == rg]
        if rg_df.empty:
            continue
        fig, axes = plt.subplots(1, len(positions), figsize=(panel_w * len(positions) + 1.5, 2.8),
                                 squeeze=False, gridspec_kw=dict(wspace=0.03))
        for ax, pos in zip(axes[0], positions):
            # Random draw order so no value systematically hides the others.
            in_slab = _in_slab(rg_df, pos, slab_um).sample(frac=1, random_state=0)
            ax.scatter(in_slab['ccf_atlas_ml'], in_slab['ccf_atlas_dv'], c=in_slab[value_col], cmap=cmap,
                       norm=norm, s=3, alpha=0.8, linewidths=0, zorder=1, rasterized=True)
            draw_atlas_outline(ax, atlas, pos, x_max)
            ax.set_title(f"AP {pos} µm\nn={len(in_slab)}", fontsize=6)
        clipped = bool((np.abs(rg_df[value_col]) > vlim).any())
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0].tolist(),
                            shrink=0.8, pad=0.01, extend='both' if clipped else 'neither')
        cbar.set_label(value_label, fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        fig.suptitle(f"{title} - {row_labels.get(rg, str(rg))} "
                     f"(n={len(rg_df)}, {rg_df['mouse_id'].nunique()} mice)", fontsize=8)
        save_figure(fig, output_folder, f"{file_prefix}_{'Rplus' if rg == 1 else 'Rminus'}{suffix}")
        plt.close(fig)


# ── one value per area group, painted on the regions ─────────────────────────

@functools.lru_cache(maxsize=2)
def _brainglobe_atlas(atlas_name=DEFAULT_ATLAS):
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
    except ImportError:
        print("[brain_slices] brainglobe_atlasapi not installed in this env "
              "(pip install brainglobe-atlasapi). Skipping brain-slice plots.")
        return None
    return BrainGlobeAtlas(atlas_name, check_latest=False)


def default_area_groups():
    """{area group: [acronyms]} from ephys_utilities.allen_utils (the lab's
    custom area groups), looked for next to this repository."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent / 'ephys_utilities'))
    from ephys_utilities.allen_utils.allen_utils import get_custom_area_groups
    return get_custom_area_groups()


def area_group_of_atlas_ids(area_groups, atlas_name=DEFAULT_ATLAS):
    """{atlas structure id: area group} for every structure belonging to a
    group: each acronym listed in area_groups and all its descendants in the
    Allen hierarchy. Where groups nest (e.g. a group lists VIS and another
    VISa), the most specific acronym wins. Acronyms that aren't Allen
    structures (the lab's custom subdivisions) are ignored, so their voxels
    fall back to whichever listed parent contains them.
    """
    atlas = _brainglobe_atlas(atlas_name)
    if atlas is None:
        return {}
    best = {}  # id -> (depth of the listed acronym, group)
    for group, acronyms in area_groups.items():
        for acronym in acronyms:
            try:
                structure = atlas.structures[acronym]
            except KeyError:
                continue
            depth = len(structure['structure_id_path'])
            ids = [structure['id']] + [atlas.structures[d]['id'] for d in atlas.get_structure_descendants(acronym)]
            for structure_id in ids:
                if structure_id not in best or depth > best[structure_id][0]:
                    best[structure_id] = (depth, group)
    return {structure_id: group for structure_id, (_, group) in best.items()}


def assign_area_group_from_coordinates(neurons, area_groups=None, atlas_name=DEFAULT_ATLAS):
    """Copy of neurons with an 'area_group' column read off the atlas at each
    neuron's CCF coordinates - the same structure-to-group map used to paint
    the regions, so every neuron is counted in the region it is drawn in.
    NaN where the position is outside the brain or in no group."""
    area_groups = area_groups or default_area_groups()
    atlas = _brainglobe_atlas(atlas_name)
    if atlas is None:
        return neurons.assign(area_group=np.nan)
    id_to_group = area_group_of_atlas_ids(area_groups, atlas_name)
    annotation = atlas.annotation  # axes (AP, DV, ML)
    valid = neurons[CCF_COLS].notna().all(axis=1).to_numpy()
    idx = [np.clip((neurons[col].fillna(0).to_numpy(dtype=float) / r).astype(int), 0, n - 1)
           for col, r, n in zip(['ccf_atlas_ap', 'ccf_atlas_dv', 'ccf_atlas_ml'], atlas.resolution, annotation.shape)]
    ids = np.asarray(annotation[idx[0], idx[1], idx[2]])
    groups = pd.Series(ids).map(id_to_group).to_numpy(dtype=object)
    groups[~valid] = np.nan
    return neurons.assign(area_group=groups)


def area_group_means_over_mice(neurons, value_col, reward_col='reward_group', group_col='area_group',
                               min_neurons_per_mouse=5, min_mice=3):
    """Mean over mice of each mouse's mean of value_col in each area group.

    A boolean value_col gives the fraction of flagged neurons per mouse, so the
    result is e.g. the proportion of selective neurons averaged over mice.
    Only mice with at least min_neurons_per_mouse neurons in the group count,
    and groups with fewer than min_mice such mice are dropped.

    Returns a DataFrame: row (one per value of reward_col found in ROW_ORDER -
    'R+' / 'R-' or the learner / expert rows - and 'pooled' for all of them),
    area_group, mean, sem, n_mice, n_neurons.
    """
    df = neurons.dropna(subset=[group_col, value_col]).copy()
    df[value_col] = df[value_col].astype(float)
    # One value per mouse and row (a mouse recorded as learner and as expert
    # contributes one value to each of those rows, and two to 'pooled').
    if reward_col in df.columns:
        df['_row'] = df[reward_col].replace({'R+': 1, 'R-': 0})
        df = df[df['_row'].isin(ROW_ORDER)]
    else:
        df['_row'] = 'pooled'
    per_mouse = (df.groupby(['mouse_id', '_row', group_col])[value_col].agg(['mean', 'size']).reset_index()
                 .rename(columns={'mean': 'value', 'size': 'n_neurons'}))
    per_mouse = per_mouse[per_mouse['n_neurons'] >= min_neurons_per_mouse]
    rows = [(row, per_mouse[per_mouse['_row'] == row]) for row in ROW_ORDER if (per_mouse['_row'] == row).any()]
    rows.append(('pooled', per_mouse))
    out = []
    for row, sub in rows:
        stats = sub.groupby(group_col).agg(mean=('value', 'mean'), sem=('value', 'sem'),
                                           n_mice=('mouse_id', 'nunique'), n_neurons=('n_neurons', 'sum'))
        stats = stats[stats['n_mice'] >= min_mice].reset_index().rename(columns={group_col: 'area_group'})
        out.append(stats.assign(row=ROW_LABELS[row]))
    return pd.concat(out, ignore_index=True)[['row', 'area_group', 'mean', 'sem', 'n_mice', 'n_neurons']]


def plot_area_group_values_on_slices(
        values, output_folder, name, label, area_groups=None,
        ap_positions=DEFAULT_AP_POSITIONS, cmap=None, vmin=None, vmax=None, scale=1.0,
        one_hemisphere=True, atlas_name=DEFAULT_ATLAS, title=None):
    """Coronal atlas slices with every area group painted in one colour: its
    value (the 2D slices of psth_per_mouse_with_barplots / brain_coefficient_viz,
    drawn straight from the atlas annotation).

    values : {area group: value} for one row of slices, or
        {row label: {area group: value}} for several rows sharing one colour
        scale (e.g. {'R+': {...}, 'R-': {...}}), or the DataFrame returned by
        area_group_means_over_mice (its row / area_group / mean columns).
    area_groups : {area group: [Allen acronyms]}; default default_area_groups().
        A group is painted over all its acronyms and their descendants.
    scale : multiplies the values before plotting (e.g. 100 for a fraction
        shown as %).
    cmap, vmin, vmax : default to a diverging map centred on 0 ('RdBu_r',
        +/- max |value|) when any value is negative, else a sequential one
        ('viridis', 0..max). Groups with no value stay white.

    Files: {output_folder}/area_groups_{name}.png/.pdf and _values.txt.
    """
    if isinstance(values, pd.DataFrame):
        values = {row: sub.set_index('area_group')['mean'].to_dict() for row, sub in values.groupby('row', sort=False)}
    elif values and not isinstance(next(iter(values.values())), dict):
        values = {'': values}
    values = {row: {g: v * scale for g, v in vals.items() if np.isfinite(v)} for row, vals in values.items()}
    values = {row: vals for row, vals in values.items() if vals}
    if not values:
        print(f"[plot_area_group_values_on_slices] No values for {name}. Skipping.")
        return
    area_groups = area_groups or default_area_groups()
    atlas = atlas_coronal_slices(tuple(ap_positions), atlas_name)
    if atlas is None or not atlas['slices']:
        return
    positions = list(atlas['slices'])
    id_to_group = area_group_of_atlas_ids(area_groups, atlas_name)

    all_vals = np.array([v for vals in values.values() for v in vals.values()])
    diverging = bool((all_vals < 0).any())
    if vmax is None:
        vmax = float(np.abs(all_vals).max()) or 1.0
    if vmin is None:
        vmin = -vmax if diverging else 0.0
    cmap = plt.get_cmap(cmap or ('RdBu_r' if diverging else 'viridis')).copy()
    cmap.set_bad(alpha=0)
    x_max = atlas['midline_um'] + 300 if one_hemisphere else atlas['width_um']
    extent = [0, atlas['width_um'], atlas['height_um'], 0]

    fig, axes = _new_figure(len(values), positions, x_max, atlas)
    for r, (row, vals) in enumerate(values.items()):
        for c, pos in enumerate(positions):
            ax = axes[r, c]
            labels = atlas['slices'][pos]['labels']
            ids, inverse = np.unique(labels, return_inverse=True)
            id_values = np.array([vals.get(id_to_group.get(int(i)), np.nan) for i in ids], dtype=float)
            ax.imshow(np.ma.masked_invalid(id_values[inverse].reshape(labels.shape)), extent=extent, cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation='nearest', zorder=1)
            draw_atlas_outline(ax, atlas, pos, x_max)
            if r == 0:
                ax.set_title(f"AP {pos} µm", fontsize=6)
        if row:
            axes[r, 0].text(-0.08, 0.5, str(row), transform=axes[r, 0].transAxes, ha='right', va='center', fontsize=8)
    clipped_low = bool((all_vals < vmin).any())
    clipped_high = bool((all_vals > vmax).any())
    extend = 'both' if clipped_low and clipped_high else ('max' if clipped_high else ('min' if clipped_low else 'neither'))
    _colorbar(fig, axes.ravel().tolist(), cmap, vmin, vmax, label, extend=extend)
    fig.suptitle(title or f"{label} per area group - {name}", fontsize=10)
    save_figure(fig, output_folder, f'area_groups_{name}')
    plt.close(fig)
    _save_table(pd.DataFrame([dict(row=row, area_group=g, value=v)
                              for row, vals in values.items() for g, v in vals.items()]),
                output_folder, f'area_groups_{name}_values.txt')


# ── ROC results ──────────────────────────────────────────────────────────────

def plot_roc_selectivity(
        roc_df, output_folder, reward_group_by_mouse=None, analysis_types=None,
        significant_col='significant', direction_col='direction', split_direction=True,
        density=True, area_group_maps=True, area_groups=None, min_neurons_per_mouse=5, min_mice=3,
        split_learning=True, id_cols=('mouse_id', 'neuron_id'), n_jobs=None,
        selectivity_maps=True, selectivity_col='selectivity', **slice_kwargs):
    """Percentage of ROC-selective neurons on atlas slices, per analysis_type.

    roc_df : ROC results (e.g. roc_analysis_utils.load_roc_results), one row
        per neuron and analysis_type, with mouse_id, neuron_id, CCF_COLS,
        analysis_type and a boolean significant_col. Filter it beforehand to
        the units you want counted (e.g. good units, areas kept).
    reward_group_by_mouse : {mouse_id: 1 or 0 (or 'R+'/'R-')}. If given (or if
        roc_df already has a reward_group column), figures have one row per
        reward group plus a pooled version; otherwise only the pooled version.
    split_learning : if roc_df has a 'learning_group' column ('learner' /
        'expert', as from load_roc_results_all_sessions) and reward groups are
        known, rows are learner R+, expert R+, learner R-, expert R- (in the
        LEARNING_COLORS colours) instead of R+ and R-.
    analysis_types : subset to plot (default: all in roc_df).
    split_direction : also map the percentage of neurons significant in each
        direction (direction_col: e.g. 'positive' / 'negative').
    density : also call plot_group_density (where the selective neurons are,
        over the sampled space).
    area_group_maps : also paint each area group with its percentage of
        significant neurons averaged over mice (area_group_means_over_mice with
        min_neurons_per_mouse / min_mice, then plot_area_group_values_on_slices).
        Uses roc_df's 'area_group' column if present, otherwise reads each
        neuron's group off the atlas at its coordinates (area_groups, default
        default_area_groups()).
    n_jobs : worker processes, one analysis type each (default: up to 6, at
        most one per analysis type; 1 = no parallelism). Each worker loads the
        atlas, about 0.5 GB of memory.
    selectivity_maps : also map the selectivity value itself (selectivity_col)
        of every neuron, significant or not: one dot per neuron and the local
        mean (plot_value_maps_on_slices), for the signed selectivity and for
        its absolute value.
    slice_kwargs : passed on (ap_positions, slab_um, sigma_um, min_neurons, ...).

    Output, under {output_folder}/{analysis_type}/: fraction/, area_groups/,
    density/, points/, density_with_sampling/ (plot_group_density) and
    selectivity/points/, selectivity/map/ (plot_value_maps_on_slices).
    """
    df = roc_df.copy()
    if reward_group_by_mouse is not None:
        df['reward_group'] = df['mouse_id'].map(reward_group_by_mouse)
    if 'reward_group' in df.columns:
        df['reward_group'] = df['reward_group'].replace({'R+': 1, 'R-': 0})
    id_cols = list(id_cols)
    if split_learning and 'learning_group' in df.columns and 'reward_group' in df.columns:
        # Rows become 'learner R+', 'expert R+', 'learner R-', 'expert R-'. The same
        # mouse is recorded in several sessions, so a neuron is (mouse, session, id).
        reward_txt = df['reward_group'].map({1: 'R+', 0: 'R-'})
        df['reward_group'] = (df['learning_group'].astype(str) + ' ' + reward_txt).where(
            reward_txt.notna() & df['learning_group'].notna())
        id_cols += [c for c in ('learning_group', 'session_id') if c in df.columns and c not in id_cols]
    density_kwargs = {k: v for k, v in slice_kwargs.items() if k not in ('min_neurons', 'vmax')}
    value_kwargs = {k: v for k, v in slice_kwargs.items() if k != 'vmax'}
    area_kwargs = {k: v for k, v in slice_kwargs.items() if k in ('ap_positions', 'atlas_name')}
    if area_group_maps and 'area_group' not in df.columns:
        # Once per neuron (the position doesn't depend on the analysis type).
        units = assign_area_group_from_coordinates(df.drop_duplicates(subset=id_cols)[id_cols + CCF_COLS],
                                                   area_groups, slice_kwargs.get('atlas_name', DEFAULT_ATLAS))
        df = df.merge(units[id_cols + ['area_group']], on=id_cols, how='left')

    # One task per analysis type; they are independent, so they can run in
    # separate processes (each loads the atlas once).
    tasks = []
    for analysis_type in (analysis_types or sorted(df['analysis_type'].dropna().unique())):
        sub = df[df['analysis_type'] == analysis_type].drop_duplicates(subset=id_cols).copy()
        if sub.empty:
            print(f"[plot_roc_selectivity] No rows for analysis_type {analysis_type!r}. Skipping.")
            continue
        tasks.append(dict(
            sub=sub, analysis_type=analysis_type, out=os.path.join(output_folder, str(analysis_type)),
            significant_col=significant_col, direction_col=direction_col, split_direction=split_direction,
            density=density, area_group_maps=area_group_maps, area_groups=area_groups,
            min_neurons_per_mouse=min_neurons_per_mouse, min_mice=min_mice, id_cols=id_cols,
            slice_kwargs=slice_kwargs, density_kwargs=density_kwargs, area_kwargs=area_kwargs,
            selectivity_maps=selectivity_maps, selectivity_col=selectivity_col, value_kwargs=value_kwargs))

    if n_jobs is None:
        n_jobs = max(1, min(6, (os.cpu_count() or 2) - 1))
    n_jobs = min(n_jobs, len(tasks))
    if n_jobs <= 1:
        for task in tasks:
            print(_plot_roc_one_analysis_type(task))
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_plot_roc_one_analysis_type, task): task['analysis_type'] for task in tasks}
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"[plot_roc_selectivity] {futures[future]} FAILED: {type(e).__name__}: {e}")


def _plot_roc_one_analysis_type(task):
    """All the figures of one analysis type (see plot_roc_selectivity); runs in
    a worker process. Returns a one-line summary."""
    matplotlib.use('Agg')  # no GUI in worker processes
    sub, analysis_type, out, id_cols = task['sub'], task['analysis_type'], task['out'], task['id_cols']
    sub['significant_any'] = sub[task['significant_col']].fillna(False).astype(bool)
    flags = {'significant': 'significant_any'}
    if task['split_direction'] and task['direction_col'] in sub.columns:
        for direction in sorted(sub.loc[sub['significant_any'], task['direction_col']].dropna().unique()):
            col = f'significant_{direction}'
            sub[col] = sub['significant_any'] & (sub[task['direction_col']] == direction)
            flags[col] = col
    for flag_name, col in flags.items():
        plot_fraction_on_slices(sub, col, os.path.join(out, 'fraction'), name=flag_name,
                                label='% of neurons', id_cols=id_cols, **task['slice_kwargs'])
    if task['density']:
        # Writes out/density, out/points and out/density_with_sampling.
        plot_group_density(sub, {flag_name: sub[sub[col]] for flag_name, col in flags.items()},
                           out, file_prefix='roc', id_cols=id_cols, **task['density_kwargs'])
    if task['selectivity_maps'] and task['selectivity_col'] in sub.columns:
        # Every neuron, significant or not. Writes out/selectivity/points and out/selectivity/map.
        sel_col, sel_out = task['selectivity_col'], os.path.join(out, 'selectivity')
        sub['selectivity_abs'] = pd.to_numeric(sub[sel_col], errors='coerce').abs()
        plot_value_maps_on_slices(sub, sel_col, sel_out, name='selectivity', label='selectivity',
                                  diverging=True, id_cols=id_cols, **task['value_kwargs'])
        plot_value_maps_on_slices(sub, 'selectivity_abs', sel_out, name='selectivity_abs',
                                  label='|selectivity|', diverging=False, id_cols=id_cols, **task['value_kwargs'])
    if task['area_group_maps']:
        area_out = os.path.join(out, 'area_groups')
        min_neurons_per_mouse, min_mice = task['min_neurons_per_mouse'], task['min_mice']
        for flag_name, col in flags.items():
            means = area_group_means_over_mice(sub, col, min_neurons_per_mouse=min_neurons_per_mouse,
                                               min_mice=min_mice)
            _save_table(means, area_out, f'area_groups_{flag_name}_mean_over_mice.txt')
            plot_area_group_values_on_slices(
                means, area_out, flag_name, '% of neurons (mean over mice)', area_groups=task['area_groups'],
                scale=100, title=f"{analysis_type} - {flag_name}: % of neurons per area group, mean over mice "
                                 f"(>= {min_neurons_per_mouse} neurons per mouse, >= {min_mice} mice)",
                **task['area_kwargs'])
    return (f"[plot_roc_selectivity] {analysis_type}: {int(sub['significant_any'].sum())}/{len(sub)} "
            f"significant -> {out}")


# ── run every ROC plot ───────────────────────────────────────────────────────
# These paths must stay above the functions below, which use them as defaults.

_MYRIAM_RESULTS = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Myriam_Hamon\combined_results_ks4"
ROC_ROOT = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\combined_results_ks4"
ROC_OUTPUT = os.path.join(_MYRIAM_RESULTS, 'roc_analysis', 'brain_slices')
MOUSE_INFO_XLSX = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Myriam_Hamon\mice_info\joint_mouse_reference_weight.xlsx"
SESSION_RATES_CSV = os.path.join(_MYRIAM_RESULTS, 'whisker_session_rates', 'whisker_session_rates_1000trials.csv')
ROC_COLUMNS = ['mouse_id', 'session_id', 'neuron_id', 'area'] + CCF_COLS + ['analysis_type', 'significant', 'direction', 'selectivity']
# Unit quality labels per filter set (qc_good_units, qc_good_mua), keyed by mouse_id,
# whisker_folder, neuron_id. Built once from the NWB unit tables by
# build_roc_unit_quality_table (needs pynwb), then read on every run.
ROC_NWB_PATH = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\NWB_ks4"
ROC_QUALITY_TABLE = os.path.join(_MYRIAM_RESULTS, 'roc_analysis', 'unit_quality_labels.csv')
# Area groups not painted in the area-group maps (same list as the GLM kernel analysis).
ROC_EXCLUDED_AREA_GROUPS = {
    'somatosensory-body', 'cortical subplate', 'amygdala and hypothalamus',
    'pons and medulla', 'insular areas', 'olfactory areas',
}


def load_reward_groups(mouse_info_xlsx=MOUSE_INFO_XLSX, session_rates_csv=SESSION_RATES_CSV):
    """{mouse_id: 1 (R+) or 0 (R-)}. From the mouse reference sheet (mice with
    exclude == 0 and recording == 1), or, if it can't be read, from the
    reward_group column of the session-rates CSV."""
    try:
        info = pd.read_excel(mouse_info_xlsx).rename(columns={'mouse_name': 'mouse_id'})
        info = info[(info['exclude'] == 0) & (info['recording'] == 1) & info['reward_group'].isin(['R+', 'R-'])]
        return info.set_index('mouse_id')['reward_group'].map({'R+': 1, 'R-': 0}).to_dict()
    except Exception as e:
        print(f"[roc] Could not read {mouse_info_xlsx} ({type(e).__name__}: {e}); "
              f"using reward groups from {session_rates_csv}.")
        rates = pd.read_csv(session_rates_csv).dropna(subset=['reward_group'])
        return rates.set_index('mouse_id')['reward_group'].astype(int).to_dict()


def load_roc_tables(root=ROC_ROOT, analysis_types=None, max_workers=8):
    """ROC results of every mouse and whisker session under root
    ({mouse}/whisker_*/roc_analysis/*_roc_results_new.csv), reading only
    ROC_COLUMNS (the files also hold the full ROC curves). Adds whisker_folder
    (whisker_0, whisker_1, ...) and learning_group: 'learner' for whisker_0,
    'expert' for later whisker days.

    As in run_roc_analysis_experts_vs_learners.py, 'positive' and 'negative'
    are swapped for the choice analysis types (they are inverted in the raw
    output)."""
    from concurrent.futures import ThreadPoolExecutor
    import glob
    import re

    files = []
    for mouse in sorted(m for m in os.listdir(root) if re.match(r'^(AB|MH)\d{3}$', m)):
        mouse_path = os.path.join(root, mouse)
        for day in sorted(d for d in os.listdir(mouse_path) if d.startswith('whisker_')):
            for path in glob.glob(os.path.join(mouse_path, day, 'roc_analysis', '*_roc_results_new.csv')):
                files.append((path, day))

    def _read(item):
        path, day = item
        try:
            df = pd.read_csv(path, usecols=lambda c: c in ROC_COLUMNS)
        except Exception as e:
            print(f"[roc] Skipped {path} ({type(e).__name__}: {e})")
            return None
        if analysis_types is not None:
            df = df[df['analysis_type'].isin(analysis_types)]
        return df.assign(whisker_folder=day, learning_group='learner' if day == 'whisker_0' else 'expert')

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tables = [t for t in pool.map(_read, files) if t is not None]
    if not tables:
        return pd.DataFrame(columns=ROC_COLUMNS + ['whisker_folder', 'learning_group'])
    roc = pd.concat(tables, ignore_index=True)
    choice = roc['analysis_type'].str.contains('choice', na=False)
    roc.loc[choice, 'direction'] = roc.loc[choice, 'direction'].replace(
        {'positive': 'negative', 'negative': 'positive'})
    return roc


def default_excluded_areas():
    """Areas left out of the ROC analyses (allen_utils.get_excluded_areas), to
    match against the ROC tables' 'area' column."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent / 'ephys_utilities'))
    from ephys_utilities.allen_utils.allen_utils import get_excluded_areas
    return list(get_excluded_areas())


def build_roc_unit_quality_table(output_csv=ROC_QUALITY_TABLE, nwb_path=ROC_NWB_PATH, motion_root=ROC_ROOT,
                                 max_workers=24):
    """Unit quality labels for every FILTER_SETS entry of
    roc_analysis/run_roc_analysis_experts_vs_learners.py, computed by that
    script's own code: NWB unit tables (combine_ephys_nwb), presence and
    coverage ratios, the drift shift-test files under motion_root, then
    classify_units_quality.

    Writes output_csv with mouse_id, whisker_folder, neuron_id and one boolean
    column qc_<filter name> per filter set, and returns it. Slow (it reads
    every NWB file) and needs pynwb and the lab's ephys utilities, so it is run
    once and cached.
    """
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for p in (here, here / 'roc_analysis'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import run_roc_analysis_experts_vs_learners as roc_script  # also sets up the ephys_utilities paths
    import neural_utils

    info = pd.read_excel(MOUSE_INFO_XLSX).rename(columns={'mouse_name': 'mouse_id'})
    mice = info.loc[info['reward_group'].isin(['R+', 'R-']), 'mouse_id'].unique()
    nwb_list = [os.path.join(nwb_path, f) for f in os.listdir(nwb_path) if any(m in f for m in mice)]
    print(f"[roc] Building unit quality labels from {len(nwb_list)} NWB files ...")
    _, unit_table, _ = neural_utils.combine_ephys_nwb(nwb_list, max_workers=max_workers)
    unit_table = roc_script.allen.process_allen_labels(unit_table)
    # whisker_N sessions have behaviour == 'whisker' and day == N; neuron_id is the
    # per-session index the ROC pipeline also recorded.
    units = unit_table[unit_table['behaviour'] == 'whisker'].copy()
    units['whisker_folder'] = 'whisker_' + units['day'].astype(int).astype(str)
    units = roc_script.add_unit_quality_labels(units, motion_root, roc_script.FILTER_SETS)
    qc_cols = [f'qc_{name}' for name in roc_script.FILTER_SETS]
    table = (units[['mouse_id', 'whisker_folder', 'neuron_id'] + qc_cols]
             .drop_duplicates(subset=['mouse_id', 'whisker_folder', 'neuron_id']))
    os.makedirs(_long_path(os.path.dirname(output_csv)), exist_ok=True)
    table.to_csv(_long_path(output_csv), index=False)
    print(f"[roc] Unit quality labels for {len(table)} units -> {output_csv}")
    return table


def load_roc_unit_quality_table(path=ROC_QUALITY_TABLE):
    """The cached table of build_roc_unit_quality_table, or None if it doesn't exist yet."""
    if not os.path.exists(_long_path(path)):
        return None
    return pd.read_csv(_long_path(path))



def main(argv=None):
    """All the ROC brain-slice plots.

    For every analysis_type (or those given with --analysis-types), in
    {out}/first_whisker_day/ (whisker_0 sessions, rows R+ / R-) and, when
    expert sessions exist, {out}/learners_vs_experts/ (all sessions, rows
    learner / expert within each reward group):
      fraction/     local % of significant neurons (all, and per direction)
      area_groups/  % of significant neurons per area group, mean over mice
      density/, points/, density_with_sampling/
                    where the significant neurons are, over the sampled space
      selectivity/  the selectivity value of every neuron (signed, and absolute):
                    points/ one dot per neuron, map/ its local mean
    Each also has a pooled version.

    Unit selection, as in run_roc_analysis_experts_vs_learners.py: units in the
    excluded areas (allen_utils.get_excluded_areas) are dropped, and everything
    is run once per unit-quality filter set (good_units, good_mua), each into
    {out}/{filter set}/. The quality labels come from a cached table
    (ROC_QUALITY_TABLE); build it once with --build-quality-table, in an
    environment that has pynwb.
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=ROC_ROOT, help='folder with one sub-folder per mouse')
    parser.add_argument('--out', default=ROC_OUTPUT, help='output folder')
    parser.add_argument('--analysis-types', nargs='+', default=['whisker_passive_pre'], help='default: every analysis_type found')
    parser.add_argument('--skip-experts', action='store_true', help='only the first-whisker-day figures')
    parser.add_argument('--skip-density', action='store_true', help='no density / points figures (faster)')
    parser.add_argument('--skip-selectivity', action='store_true',
                        help='no selectivity-value figures (points and local mean)')
    parser.add_argument('--skip-2000um', action='store_true',
                        help='no second version with 2000 um slabs at AP 3500, 5000, 7000, 9000')
    parser.add_argument('--min-mice', type=int, default=3, help='mice needed to paint an area group')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='worker processes, one analysis type each (default: up to 6; 1 = sequential)')
    parser.add_argument('--filter-sets', nargs='+', default=None,
                        help='unit-quality filter sets to run (default: all in the quality table)')
    parser.add_argument('--quality-table', default=ROC_QUALITY_TABLE, help='cached unit quality labels (csv)')
    parser.add_argument('--build-quality-table', action='store_true',
                        help='only (re)build the unit quality table from the NWB files, then stop; needs pynwb')
    args = parser.parse_args(argv)
    import sys

    if args.build_quality_table:
        build_roc_unit_quality_table(args.quality_table)
        return

    # Every figure needs the atlas: stop now rather than load everything and plot nothing.
    if _brainglobe_atlas() is None:
        raise SystemExit(f"[roc] No figures can be made with this Python ({sys.executable}): "
                         f"brainglobe_atlasapi is missing. Install it (pip install brainglobe-atlasapi) "
                         f"or run with an environment that has it, e.g. brainrender or cicada_env.")

    quality = load_roc_unit_quality_table(args.quality_table)
    if quality is None:
        print(f"[roc] No unit quality table at {args.quality_table}; building it (reads every NWB file) ...")
        try:
            quality = build_roc_unit_quality_table(args.quality_table)
        except ImportError as e:
            raise SystemExit(
                f"[roc] Could not build the unit quality table with this Python ({sys.executable}): {e}.\n"
                f"Build it once in an environment that has pynwb, then run again here:\n"
                f"    <python of that environment> visualize_quantity.py --build-quality-table")
    qc_cols = [c for c in quality.columns if c.startswith('qc_')]
    filter_sets = args.filter_sets or [c[3:] for c in qc_cols]
    unknown = [f for f in filter_sets if f'qc_{f}' not in quality.columns]
    if unknown:
        raise SystemExit(f"[roc] Filter sets {unknown} are not in the quality table (it has {qc_cols}).")

    t0 = time.time()
    reward_groups = load_reward_groups()
    roc = load_roc_tables(args.root, args.analysis_types)
    if roc.empty:
        print(f"[roc] No ROC results found under {args.root}.")
        return
    no_group = sorted(set(roc['mouse_id']) - set(reward_groups))
    if no_group:
        print(f"[roc] {len(no_group)} mice without a reward group are left out: {no_group}")
    roc = roc[roc['mouse_id'].isin(reward_groups)]
    print(f"[roc] Loaded {roc['mouse_id'].nunique()} mice in {time.time() - t0:.0f}s:")
    print(roc.assign(reward_group=roc['mouse_id'].map(reward_groups).map({1: 'R+', 0: 'R-'}))
          .groupby(['learning_group', 'reward_group'])
          .agg(mice=('mouse_id', 'nunique'), sessions=('session_id', 'nunique')).to_string())

    # Unit selection: excluded areas out, then the quality label of each filter set.
    # Units with no quality label can't be vetted, so they fail every filter set.
    unit_key = ['mouse_id', 'whisker_folder', 'neuron_id']
    n_units = len(roc.drop_duplicates(unit_key))
    roc = roc[~roc['area'].isin(default_excluded_areas())]
    n_after_areas = len(roc.drop_duplicates(unit_key))
    roc = roc.merge(quality[unit_key + qc_cols].drop_duplicates(unit_key), on=unit_key, how='left')
    n_labelled = int(roc.drop_duplicates(unit_key)[qc_cols[0]].notna().sum())
    roc[qc_cols] = roc[qc_cols].fillna(False).astype(bool)
    units = roc.drop_duplicates(unit_key)
    print(f"[roc] Units: {n_units} in the ROC files, {n_after_areas} outside the excluded areas, "
          f"{n_labelled} with a quality label; passing: "
          + ", ".join(f"{f} {int(units[f'qc_{f}'].sum())}" for f in filter_sets))

    area_groups = {g: a for g, a in default_area_groups().items() if g.lower() not in ROC_EXCLUDED_AREA_GROUPS}
    common = dict(reward_group_by_mouse=reward_groups, analysis_types=args.analysis_types,
                  density=not args.skip_density, selectivity_maps=not args.skip_selectivity,
                  area_groups=area_groups, min_mice=args.min_mice, n_jobs=args.n_jobs)

    # Slice settings: the default 1000 um slabs, then the custom 2000 um slabs in a
    # 'slab_2000um' sub-folder (same positions as the GLM selectivity maps). The
    # area-group maps don't depend on slab thickness, so they are made once.
    slice_sets = [('', dict())]
    if not args.skip_2000um:
        slice_sets.append(('slab_2000um', dict(ap_positions=(3500, 5000, 7000, 9000), slab_um=2000,
                                               area_group_maps=False)))

    for filter_set in filter_sets:
        kept = roc[roc[f'qc_{filter_set}']]
        if kept.empty:
            print(f"[roc] No units pass filter set {filter_set!r}. Skipping.")
            continue
        out = os.path.join(args.out, filter_set)
        print(f"[roc] Filter set {filter_set!r} -> {out}")
        learners = kept[kept['learning_group'] == 'learner']
        for sub_folder, slice_settings in slice_sets:
            if not learners.empty:
                plot_roc_selectivity(learners, os.path.join(out, 'first_whisker_day', sub_folder),
                                     split_learning=False, id_cols=('mouse_id', 'session_id', 'neuron_id'),
                                     **common, **slice_settings)
            if not args.skip_experts and (kept['learning_group'] == 'expert').any():
                plot_roc_selectivity(kept, os.path.join(out, 'learners_vs_experts', sub_folder),
                                     split_learning=True, **common, **slice_settings)
    print(f"[roc] Done in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == '__main__':
    main()
