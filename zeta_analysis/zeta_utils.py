#! /usr/bin/env/python3
"""
@project: unit_spikes_analysis
@file: zeta_analysis/zeta_utils.py
@time: 2026-11-18

ZETA-test counterpart of roc_analysis/roc_utils_new.compute_unit_roc: same units, same trial selection, same
analysis types and per-neuron labels, one row per neuron x analysis type, but responsiveness / condition differences
are tested with ZETA (Montijn et al. 2021 eLife; zetapy) and latencies estimated with latenZy (Haak & Heimel 2025
J Neurophysiol; github.com/Herseninstituut/latenZy).

Per analysis type (the ROC list of roc_utils_new.select_spike_counts, + whisker_miss_active / auditory_miss_active):
  one-condition types (event-evoked response: *_passive_pre/post, *_active, *_miss_active, spontaneous_licks)
      zetatest (vs a constant rate, jitter null) on two windows for stimuli: -200..200 ms (zeta_p_full; the
        pre-event part makes a rate step at the event that stays flat afterwards detectable) and 0..200 ms
        (zeta_p_post; more power for short transients); zeta_p = min(1, 2 * min of the two) (Bonferroni).
        Spontaneous licks: one window -200..200 ms.
      direction  = sign of the largest IFR deviation from the pre-event rate after the event ('positive' / 'negative')
      ifr_peak_s = time of the largest IFR deviation (in the direction above); pos_peak_* = largest increase
      neg_peak   = early dip: IFR minimum over 5..150 ms, at <= 100 ms, >= NEG_PEAK_MIN_Z baseline-IFR SDs and >= NEG_PEAK_MIN_DROP
                   below the pre-event rate, in a ZETA-significant neuron (any dominant direction; stimulus events)
      latency_s  = latenZy onset (window with a pre-event part), NaN when latenZy finds no significant onset;
                   run for every tested neuron (LATENZY_ALL), gate with zeta_p / latenzy_p
  two-condition types (ROC contrasts, condition 1 = spikes_1, condition 2 = spikes_2 of the ROC)
      zetatest2 (difference of the two cumulative spike-time curves vs a trial-swap null) -> zeta_p, zeta_z
      direction  = 'positive' if the mean rate in the window is higher in condition 2 (ROC convention; for
                   wh_vs_aud_* 'auditory' / 'whisker' as in the ROC)
      latency_s  = latenZy2 divergence time (every tested neuron if LATENZY_ALL)
Windows: stimulus events -200..200 ms (one-condition) / 0..200 ms (two-condition) (latenZy -100..200 ms, negative latencies not allowed); spontaneous licks
  -200..200 ms around the lick (latenZy -200..200 ms); two-condition types 0..200 ms after each condition's event
  (lick: 0..200 ms after the lick, as the ROC post window); baseline_* types -1000..-15 ms before the event (ROC).
Spikes: whisker artefact window replaced as in the ROC (correct_neuron_spike_train), seeded per neuron.
Outputs (results_path, i.e. <mouse>/<behaviour>_<day>/zeta/):
  <mouse>_zeta_results.csv   one row per neuron x analysis_type (+ ROC-style neuron labels from ZETA significance)
  <mouse>_zeta_ifr.npz       IFR of every neuron for the one-condition types on a 1-ms grid
  <mouse>_zeta_config.json   run parameters
  <mouse>_zeta_summary.pdf   ZETA-only session summary
"""

import datetime
import json
import multiprocessing
import os
import pathlib
import warnings
import zlib

import numpy as np
import pandas as pd

import NWB_reader_functions as nwb_reader
from roc_analysis import roc_utils_new as ru

ALPHA = 0.05
N_RESAMPLING = 100
N_WORKERS = 100
MIN_EVENTS = 3
STIM_WINDOW = (0.0, 0.200)                       # two-condition stimulus windows (ROC post window side)
STIM_WINDOW_ONE = (-0.200, 0.200)                # one-condition ZETA window: includes the pre-event period, so a
                                                 # rate step at the event that stays flat afterwards is detected
LICK_WINDOW = (-0.200, 0.200)
BASELINE_WINDOW = ru.BASELINE_WINDOW            # (-1.0, -0.015)
LATENZY_STIM_DUR = [-0.100, 0.200]
LATENZY_LICK_DUR = [-0.200, 0.200]
IFR_PEAK_MIN_S = 0.005                          # ignore the whisker artefact-replaced window for peak times
IFR_EDGE_S = 0.020                              # IFR within 20 ms of a window edge excluded from peak / direction
                                                # search (the multi-scale derivative is biased at the edges)
IFR_BASE_WINDOW = (-0.180, -0.010)              # IFR baseline (mean / SD) for peak z-scores: before the artefact window
NEG_PEAK_MAX_S = 0.100                          # early negative peak (dip): IFR minimum over IFR_PEAK_MIN_S..
NEG_PEAK_SEARCH_S = 0.150                       # NEG_PEAK_SEARCH_S that falls before NEG_PEAK_MAX_S (a minimum at the
                                                # end of a shorter search window would be a sustained suppression)
NEG_PEAK_MIN_Z = 3.0                            # dip >= 3 baseline-IFR SDs below the baseline mean
NEG_PEAK_MIN_DROP = 0.25                        # and >= 25 % below the baseline mean
LATENZY_ALL = True                              # run latenZy / latenZy2 for every tested neuron, not only ZETA p < ALPHA
IFR_GRID_STEP = 0.001

# (event, context, alignment); alignment in {'stim', 'lick', 'baseline'}
ONE_CONDITION = {
    'whisker_passive_pre': ('whisker', 'passive_pre', 'stim'),
    'whisker_passive_post': ('whisker', 'passive_post', 'stim'),
    'whisker_active': ('whisker', 'active', 'stim'),
    'whisker_miss_active': ('whisker_miss', 'active', 'stim'),
    'auditory_passive_pre': ('auditory', 'passive_pre', 'stim'),
    'auditory_passive_post': ('auditory', 'passive_post', 'stim'),
    'auditory_active': ('auditory', 'active', 'stim'),
    'auditory_miss_active': ('auditory_miss', 'active', 'stim'),
    'spontaneous_licks': ('spontaneous_licks', '', 'lick'),
}
TWO_CONDITION = {   # condition 1 = ROC spikes_1, condition 2 = ROC spikes_2
    'whisker_pre_vs_post_learning': (('whisker', 'passive_pre', 'stim'), ('whisker', 'passive_post', 'stim')),
    'auditory_pre_vs_post_learning': (('auditory', 'passive_pre', 'stim'), ('auditory', 'passive_post', 'stim')),
    'wh_vs_aud_passive_pre': (('whisker', 'passive_pre', 'stim'), ('auditory', 'passive_pre', 'stim')),
    'wh_vs_aud_passive_post': (('whisker', 'passive_post', 'stim'), ('auditory', 'passive_post', 'stim')),
    'wh_vs_aud_active': (('whisker', 'active', 'stim'), ('auditory', 'active', 'stim')),
    'wh_vs_aud_pre_vs_post_learning': (('whisker', 'passive_pre', 'stim'), ('auditory', 'passive_post', 'stim')),
    'spontaneous_licks_vs_cr': (('correct_rejection', 'active', 'stim'), ('spontaneous_licks', '', 'lick')),
    'choice': (('lick_trial', 'active', 'stim'), ('no_lick_trial', 'active', 'stim')),
    'whisker_choice': (('whisker_hit', 'active', 'stim'), ('whisker_miss', 'active', 'stim')),
    'auditory_choice': (('auditory_hit', 'active', 'stim'), ('auditory_miss', 'active', 'stim')),
    'whisker_sensory': (('correct_rejection', 'active', 'stim'), ('whisker_miss', 'active', 'stim')),
    'auditory_sensory': (('correct_rejection', 'active', 'stim'), ('auditory_miss', 'active', 'stim')),
    'whisker_hit_vs_cr': (('correct_rejection', 'active', 'stim'), ('whisker_hit', 'active', 'stim')),
    'auditory_hit_vs_cr': (('correct_rejection', 'active', 'stim'), ('auditory_hit', 'active', 'stim')),
    'whisker_hit_vs_spontaneous': (('spontaneous_licks', '', 'lick'), ('whisker_hit', 'active', 'stim')),
    'auditory_hit_vs_spontaneous': (('spontaneous_licks', '', 'lick'), ('auditory_hit', 'active', 'stim')),
    'baseline_choice': (('lick_trial', 'active', 'baseline'), ('no_lick_trial', 'active', 'baseline')),
    'baseline_whisker_choice': (('whisker_hit', 'active', 'baseline'), ('whisker_miss', 'active', 'baseline')),
    'baseline_auditory_choice': (('auditory_hit', 'active', 'baseline'), ('auditory_miss', 'active', 'baseline')),
    'baseline_pre_vs_post_learning': ((('whisker', 'auditory'), 'passive_pre', 'baseline'),
                                      (('whisker', 'auditory'), 'passive_post', 'baseline')),
}
PASSIVE_TYPES = [t for t, v in {**ONE_CONDITION, **TWO_CONDITION}.items() if 'passive' in str(v)]
ONE_COLORS = {'whisker': ru.WHISKER_COLOR, 'auditory': ru.AUDITORY_COLOR, 'spontaneous': '#4daf4a'}

_G = {}  # per-worker globals (set before forking)


# ------------------------------------------------------------------------------------------------ events
def event_times(trials, lick_times, event, context, has_context):
    """Event (trial start) times, the same selection as roc_utils_new.extract_event_times."""
    if isinstance(event, tuple):
        return np.sort(np.concatenate([event_times(trials, lick_times, e, context, has_context) for e in event]))
    if event == 'spontaneous_licks':
        return np.asarray(lick_times, float)
    t = trials
    cond = {
        'whisker': t['whisker_stim'] == 1,
        'auditory': t['auditory_stim'] == 1,
        'lick_trial': t['lick_flag'] == 1,
        'no_lick_trial': t['lick_flag'] == 0,
        'whisker_hit': (t['whisker_stim'] == 1) & (t['lick_flag'] == 1),
        'whisker_miss': (t['whisker_stim'] == 1) & (t['lick_flag'] == 0),
        'auditory_hit': (t['auditory_stim'] == 1) & (t['lick_flag'] == 1),
        'auditory_miss': (t['auditory_stim'] == 1) & (t['lick_flag'] == 0),
        'correct_rejection': (t['whisker_stim'] == 0) & (t['auditory_stim'] == 0) & (t['lick_flag'] == 0),
    }[event]
    if has_context:
        cond &= t['context'] == context
    elif context.startswith('passive'):
        return np.array([])
    return np.sort(t.loc[cond, 'start_time'].to_numpy(float))


def aligned(ev, alignment, one_condition=False):
    """(event times shifted to the window start, window duration, offset to add back to get event-relative times)"""
    if alignment == 'baseline':
        return ev + BASELINE_WINDOW[0], BASELINE_WINDOW[1] - BASELINE_WINDOW[0], BASELINE_WINDOW[0]
    if alignment == 'lick' and one_condition:
        return ev + LICK_WINDOW[0], LICK_WINDOW[1] - LICK_WINDOW[0], LICK_WINDOW[0]
    if one_condition:
        return ev + STIM_WINDOW_ONE[0], STIM_WINDOW_ONE[1] - STIM_WINDOW_ONE[0], STIM_WINDOW_ONE[0]
    return ev + STIM_WINDOW[0], STIM_WINDOW[1] - STIM_WINDOW[0], STIM_WINDOW[0]


def window_rate(st, ev, t0, t1):
    a = np.searchsorted(st, ev + t0)
    b = np.searchsorted(st, ev + t1)
    return (b - a).mean() / (t1 - t0) if len(ev) else np.nan


# ------------------------------------------------------------------------------------------------ per task
def _patch_latenzy():
    """latenZy 1.1.1 bug: latenzy2 shrinks its window iteratively; once a condition has no spike left in the window,
    dependencies.get_distinct_spikes(np.array([])) raises (np.max of an empty array) instead of letting
    calc_temp_diff2 handle the empty condition (it pads empty spike vectors) and latenzy2 stop at its
    'len(real_diff) < 3' guard. Patched here (not in site-packages): an empty input is returned unchanged."""
    import dependencies as lz_dep              # latenZy's helper module (installed top-level as 'dependencies')
    if not hasattr(lz_dep, 'calc_temp_diff2') or getattr(lz_dep.get_distinct_spikes, '_empty_safe', False):
        return
    orig = lz_dep.get_distinct_spikes

    def get_distinct_spikes(spike_times):
        spike_times = np.asarray(spike_times, float)
        return spike_times if spike_times.size == 0 else orig(spike_times)

    get_distinct_spikes._empty_safe = True
    lz_dep.get_distinct_spikes = get_distinct_spikes


def _retry(fun, seed, n_tries=3):
    """latenZy / latenZy2 resampling can draw only spike-less trials in a shrunken window and raise; retry with a new
    seed (up to n_tries). fun returns latenZy's (latency, info). Returns (latency in s or NaN = no significant
    onset found by latenZy, latenZy p-value of the peak that set the latency, error of the last failure or None)."""
    err = None
    for k in range(n_tries):
        np.random.seed((seed + k) % 2 ** 32)
        try:
            lat, info = fun()
            if lat is None or not np.isfinite(lat):
                return np.nan, np.nan, None
            pv = np.asarray(info.get('pValsPeak', []), float).ravel()
            ix = np.asarray(info.get('latenzyIdx', []), bool).ravel()
            lp = float(pv[ix][0]) if ix.size == pv.size and ix.any() else np.nan
            return float(lat), lp, None
        except Exception as e:
            err = f'{type(e).__name__}: {e}'[:120]
    return np.nan, np.nan, err


def _one_condition(st, ev, alignment, seed):
    from zetapy import zetatest
    from latenzy import latenzy
    _patch_latenzy()
    ev_s, dur, off = aligned(ev, alignment, one_condition=True)
    out = zetatest(st, ev_s, max_duration=dur, resampling_number=N_RESAMPLING, return_rate=True, random_seed=seed)
    p_full, zd, rd = out[:3]
    p_post, z_post = np.nan, np.nan
    if alignment == 'stim':
        # second window 0..200 ms (post-event only): more power for short transients; combined p = Bonferroni(2)
        o2 = zetatest(st, ev + STIM_WINDOW[0], max_duration=STIM_WINDOW[1] - STIM_WINDOW[0],
                      resampling_number=N_RESAMPLING, random_seed=seed)
        p_post, z_post = float(o2[0]), float(o2[1].get('zeta_score', np.nan))
        p = min(1.0, 2 * np.nanmin([p_full, p_post]))
    else:
        p = p_full
    base = window_rate(st, ev, -0.2 if alignment == 'stim' else -0.4, 0.0 if alignment == 'stim' else -0.2)
    res = dict(zeta_p=float(p), zeta_p_full=float(p_full), zeta_p_post=p_post,
               zeta_z=float(zd.get('zeta_score', np.nan)), zeta_z_post=z_post,
               zeta_time_s=float(zd['latency_zeta']) + off if zd.get('latency_zeta') is not None else np.nan,
               mean_rate_1=base, mean_rate_2=window_rate(st, ev, max(off, 0.0), off + dur),
               direction=np.nan, ifr_peak_s=np.nan, ifr_peak_hz=np.nan,
               pos_peak_s=np.nan, pos_peak_hz=np.nan, pos_peak_z=np.nan,
               neg_peak_s=np.nan, neg_peak_hz=np.nan, neg_peak_z=np.nan, neg_peak_drop=np.nan, neg_peak=False,
               latency_s=np.nan, latenzy_p=np.nan)
    grid = np.arange(off, off + dur + 1e-9, IFR_GRID_STEP)
    ifr = np.full(len(grid), np.nan, np.float32)
    if isinstance(rd, dict) and rd.get('rate_vector') is not None and len(rd['rate_vector']) > 1:
        tv, rv = np.asarray(rd['timestamps']) + off, np.asarray(rd['rate_vector'])
        ifr = np.interp(grid, tv, rv).astype(np.float32)   # edge values held before the first / after the last spike
        g = grid.astype(float)
        b = (g >= IFR_BASE_WINDOW[0]) & (g < IFR_BASE_WINDOW[1])
        sd = float(np.std(ifr[b])) if (alignment == 'stim' and b.sum() > 10) else np.nan
        k = (tv >= (IFR_PEAK_MIN_S if alignment == 'stim' else off + IFR_EDGE_S)) & (tv <= off + dur - IFR_EDGE_S)
        if k.any():
            d = rv[k] - base
            up = d.max() >= -d.min()
            i = np.argmax(d) if up else np.argmin(d)
            res.update(direction='positive' if up else 'negative', ifr_peak_s=float(tv[k][i]), ifr_peak_hz=float(rv[k][i]))
            j = np.argmax(d)
            res.update(pos_peak_s=float(tv[k][j]), pos_peak_hz=float(rv[k][j]),
                       pos_peak_z=float(d[j] / sd) if sd > 0 else np.nan)
        if alignment == 'stim':
            e = (g >= IFR_PEAK_MIN_S) & (g <= NEG_PEAK_SEARCH_S)
            j = np.argmin(ifr[e])
            dip = float(ifr[e][j]) - base
            z = dip / sd if sd > 0 else np.nan
            drop = -dip / base if base > 0 else np.nan
            res.update(neg_peak_s=float(g[e][j]), neg_peak_hz=float(ifr[e][j]), neg_peak_z=z, neg_peak_drop=drop,
                       neg_peak=bool(p < ALPHA and g[e][j] <= NEG_PEAK_MAX_S and np.isfinite(z) and -z >= NEG_PEAK_MIN_Z
                                     and drop >= NEG_PEAK_MIN_DROP))
    if p < ALPHA or LATENZY_ALL:
        dur_l = LATENZY_STIM_DUR if alignment == 'stim' else LATENZY_LICK_DUR
        lat, lp, err = _retry(lambda: latenzy(st, ev, use_dur=dur_l, resamp_num=N_RESAMPLING,
                                              restrict_neg=(alignment == 'stim')), seed)
        res.update(latency_s=lat, latenzy_p=lp)
        if err:
            res['latency_error'] = err
    return res, ifr


def _two_condition(st, ev1, al1, ev2, al2, seed, analysis_type):
    from zetapy import zetatest2
    from latenzy import latenzy2
    _patch_latenzy()
    e1, dur, off = aligned(ev1, al1)
    e2, _, _ = aligned(ev2, al2)
    np.random.seed(seed % 2 ** 32)
    out = zetatest2(st, e1, st, e2, max_duration=dur, resampling_number=N_RESAMPLING)
    p, zd = out[0], out[1]
    r1, r2 = window_rate(st, e1, 0, dur), window_rate(st, e2, 0, dur)
    dirs = ('auditory', 'whisker') if 'wh_vs_aud' in analysis_type else ('positive', 'negative')
    res = dict(zeta_p=float(p), zeta_z=float(zd.get('zeta_score', np.nan)),
               zeta_time_s=float(zd['zeta_time']) + off if zd.get('zeta_time') is not None else np.nan,
               mean_rate_1=r1, mean_rate_2=r2, direction=dirs[0] if r2 >= r1 else dirs[1],
               ifr_peak_s=np.nan, ifr_peak_hz=np.nan, latency_s=np.nan, latenzy_p=np.nan)
    if p < ALPHA or LATENZY_ALL:
        lat, lp, err = _retry(lambda: latenzy2(st, e1, st, e2, use_dur=[0, dur], resamp_num=N_RESAMPLING), seed)
        res.update(latency_s=lat + off, latenzy_p=lp)
        if err:
            res['latency_error'] = err
    return res


def _unit_task(i):
    """all analysis types for neuron i (index into _G['units'])"""
    warnings.simplefilter('ignore')
    import logging
    logging.disable(logging.WARNING)
    u = _G['units'][i]
    st = u['spike_times']
    rows, ifrs = [], {}
    for at in _G['types']:
        seed = zlib.crc32(f"{u['session_id']}_{u['neuron_id']}_{at}".encode())
        base = {k: u[k] for k in _G['meta_cols']}
        base.update(analysis_type=at)
        if at in ONE_CONDITION:
            ev_name, ctx, al = ONE_CONDITION[at]
            ev = _G['events'][(ev_name, ctx)]
            base.update(n_events_1=len(ev), n_events_2=np.nan)
            if len(ev) < MIN_EVENTS or len(st) < 3:
                rows.append({**base, 'zeta_p': np.nan, 'significant': False})
                continue
            try:
                r, ifr = _one_condition(st, ev, al, seed)
            except Exception as e:
                rows.append({**base, 'zeta_p': np.nan, 'significant': False, 'error': f'{type(e).__name__}: {e}'[:120]})
                continue
            ifrs[at] = ifr
        else:
            (n1, c1, a1), (n2, c2, a2) = TWO_CONDITION[at]
            ev1, ev2 = _G['events'][(n1, c1)], _G['events'][(n2, c2)]
            base.update(n_events_1=len(ev1), n_events_2=len(ev2))
            if min(len(ev1), len(ev2)) < MIN_EVENTS or len(st) < 3:
                rows.append({**base, 'zeta_p': np.nan, 'significant': False})
                continue
            try:
                r = _two_condition(st, ev1, a1, ev2, a2, seed, at)
            except Exception as e:
                rows.append({**base, 'zeta_p': np.nan, 'significant': False, 'error': f'{type(e).__name__}: {e}'[:120]})
                continue
        r['significant'] = bool(r['zeta_p'] < ALPHA)
        if not r['significant']:
            r['direction'] = np.nan
        rows.append({**base, **r})
    return rows, ifrs


# ------------------------------------------------------------------------------------------------ driver
def _build_config(nwb_file, mouse_id, types, counts):
    return {
        'source_nwb_file': str(nwb_file), 'mouse_id': mouse_id,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'method': 'zetapy.zetatest / zetatest2 (ZETA), latenzy.latenzy / latenzy2 (latenZy)',
        'packages': {'zetapy': '4.2', 'latenzy': '1.1.1 (Herseninstituut/latenZy@a2e89717)'},
        'n_resampling': N_RESAMPLING, 'alpha': ALPHA, 'min_events': MIN_EVENTS,
        'multiple_comparisons_correction': None, 'latenzy_all_neurons': LATENZY_ALL,
        'negative_peak': {'window_s': [IFR_PEAK_MIN_S, NEG_PEAK_MAX_S], 'search_window_s': [IFR_PEAK_MIN_S, NEG_PEAK_SEARCH_S], 'min_z': NEG_PEAK_MIN_Z,
                          'min_drop': NEG_PEAK_MIN_DROP, 'requires_zeta_p_below_alpha': True},
        'artefact_correction': 'roc_utils_new.correct_neuron_spike_train (whisker onsets, %s s replaced by Poisson '
                               'spikes at the %s s rate), seeded per neuron' % (list(ru.WHISKER_ARTIFACT_WINDOW),
                                                                               list(ru.WHISKER_ARTIFACT_BASELINE_WINDOW)),
        'windows_sec': {'stim_one_condition': list(STIM_WINDOW_ONE), 'stim_two_condition': list(STIM_WINDOW),
                        'ifr_baseline': list(IFR_BASE_WINDOW), 'ifr_peak_min': IFR_PEAK_MIN_S, 'ifr_edge_excluded': IFR_EDGE_S, 'spontaneous_licks_one_condition': list(LICK_WINDOW),
                        'baseline': list(BASELINE_WINDOW), 'latenzy_stim': LATENZY_STIM_DUR,
                        'latenzy_lick': LATENZY_LICK_DUR},
        'analyses_to_do': types, 'n_events': {f'{k[0]}|{k[1]}': int(v) for k, v in counts.items()},
        'definitions': {
            'one_condition': 'stimuli: zetatest on -200..200 ms and 0..200 ms, zeta_p = Bonferroni(2) min; licks: '
                             '-200..200 ms; direction = sign of largest IFR deviation from '
                             'the pre-event rate (-200..0 ms; licks -400..-200 ms); latency_s = latenZy onset '
                             '(stim: -100..200 ms, no negative latencies; licks -200..200 ms, negative = before the lick); NaN = no '
                             'significant onset; neg_peak = early dip, see negative_peak',
            'two_condition': 'zetatest2 between condition 1 (ROC spikes_1) and 2 (ROC spikes_2), each aligned to its '
                             'own event; direction positive if rate higher in condition 2; latency_s = latenZy2 '
                             'divergence time (>= 0, after each condition event; NaN = none)',
            'trial_selection': 'as roc_utils_new.extract_event_times (process_nwb_tables contexts; filtered '
                               'spontaneous licks); no perf / disengagement filtering (as the ROC)',
            'labels': 'roc_utils_new.compute_neuron_labels applied to ZETA significance / direction',
        },
    }


def load_session(nwb_file):
    """units (artefact-corrected spike trains, seeded per neuron) + event times + analysis types, as for the ROC"""
    nwb_file = pathlib.Path(nwb_file)
    mouse_id, session_id = nwb_file.name[:5], nwb_file.stem
    nwb = nwb_reader.read_nwb_file(nwb_file)
    units, trials = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
    lick_times = ru.get_filtered_lick_times(nwb)
    contexts = trials['context'].unique()
    has_context = 'active' in contexts and 'passive_pre' in contexts

    events = {}
    for spec in list(ONE_CONDITION.values()) + [c for pair in TWO_CONDITION.values() for c in pair]:
        key = (spec[0], spec[1])
        if key not in events:
            events[key] = event_times(trials, lick_times, spec[0], spec[1], has_context)
    types = [t for t in list(ONE_CONDITION) + list(TWO_CONDITION) if has_context or t not in PASSIVE_TYPES]

    whisker_onsets = trials.loc[trials['whisker_stim'] == 1, 'start_time'].to_numpy()
    units['mouse_id'], units['session_id'] = mouse_id, session_id
    ccf_cols = [c for c in units.columns if 'ccf' in c]
    meta_cols = ['mouse_id', 'session_id', 'electrode_group', 'neuron_id', 'cluster_id', 'firing_rate',
                 'target_region'] + ccf_cols
    U = []
    for _, u in units.iterrows():
        rng = np.random.default_rng(zlib.crc32(f'{session_id}_{u.neuron_id}'.encode()))
        d = {k: u[k] for k in meta_cols}
        d['spike_times'] = np.sort(ru.correct_neuron_spike_train(np.asarray(u['spike_times'], float), whisker_onsets, rng))
        U.append(d)
    return mouse_id, session_id, U, events, types, meta_cols


def compute_unit_zeta(nwb_file, results_path, n_workers=None, make_summary=True):
    """ZETA / latenZy counterpart of roc_utils_new.compute_unit_roc for one NWB file (see module docstring)."""
    n_workers = n_workers or N_WORKERS
    mouse_id, session_id, U, events, types, meta_cols = load_session(nwb_file)
    _G.update(units=U, events=events, types=types, meta_cols=meta_cols)
    print(f'{session_id}: {len(U)} units x {len(types)} analysis types, {n_workers} workers')

    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'
    if n_workers > 1:
        with multiprocessing.get_context('fork').Pool(n_workers, maxtasksperchild=200) as pool:
            out = pool.map(_unit_task, range(len(U)), chunksize=max(1, len(U) // (n_workers * 4)))
    else:
        out = [_unit_task(i) for i in range(len(U))]

    results = pd.DataFrame([r for rows, _ in out for r in rows])
    results['area'] = results['ccf_atlas_parent_acronym'] if 'ccf_atlas_parent_acronym' in results else np.nan
    results['p_value'] = results['zeta_p']
    results = results.merge(ru.compute_neuron_labels(results), on='neuron_id', how='left')

    os.makedirs(results_path, exist_ok=True)
    results.to_csv(os.path.join(results_path, f'{mouse_id}_zeta_results.csv'), index=False)
    ifr_save = {}
    for at in ONE_CONDITION:
        if at not in types:
            continue
        al = ONE_CONDITION[at][2]
        ok = [(U[i]['neuron_id'], o[1][at]) for i, o in enumerate(out) if at in o[1]]
        if ok:
            w = LICK_WINDOW if al == 'lick' else STIM_WINDOW_ONE
            off, dur = w[0], w[1] - w[0]
            ifr_save[f'{at}__neuron_id'] = np.array([n for n, _ in ok])
            ifr_save[f'{at}__ifr'] = np.stack([x for _, x in ok])
            ifr_save[f'{at}__t_s'] = np.arange(off, off + dur + 1e-9, IFR_GRID_STEP)
    np.savez_compressed(os.path.join(results_path, f'{mouse_id}_zeta_ifr.npz'), **ifr_save)
    with open(os.path.join(results_path, f'{mouse_id}_zeta_config.json'), 'w') as f:
        json.dump(_build_config(nwb_file, mouse_id, types, {k: len(v) for k, v in events.items()}), f, indent=2)
    if make_summary:
        plot_zeta_summary(results, ifr_save, U, events, os.path.join(results_path, f'{mouse_id}_zeta_summary.pdf'),
                          session_id)
    print('Saved ZETA results to', results_path)
    return results


def make_zeta_summary(nwb_file, results_path):
    """rebuild <mouse>_zeta_summary.pdf from the saved CSV / IFR files (spike trains reloaded for the rasters)"""
    mouse_id, session_id, U, events, _, _ = load_session(nwb_file)
    results = pd.read_csv(os.path.join(results_path, f'{mouse_id}_zeta_results.csv'), low_memory=False)
    ifr = dict(np.load(os.path.join(results_path, f'{mouse_id}_zeta_ifr.npz')))
    plot_zeta_summary(results, ifr, U, events, os.path.join(results_path, f'{mouse_id}_zeta_summary.pdf'), session_id)


# ------------------------------------------------------------------------------------------------ summary PDF
def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 7, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']})
    return plt


def _type_color(at):
    for k, c in ONE_COLORS.items():
        if at.startswith(k):
            return c
    return '0.4'


def plot_zeta_summary(results, ifr, units, events, pdf_path, session_id):
    """ZETA-only session summary: fractions, latencies, per-area fractions, IFR heatmaps, example neurons."""
    plt = _style()
    from matplotlib.backends.backend_pdf import PdfPages
    R = results.copy()
    n_units = R.neuron_id.nunique()
    types = [t for t in list(ONE_CONDITION) + list(TWO_CONDITION) if t in set(R.analysis_type)]
    one = [t for t in ONE_CONDITION if t in types and R.loc[R.analysis_type == t, 'zeta_p'].notna().any()]
    with PdfPages(pdf_path) as pdf:
        # page 1: fraction significant per analysis type, split by direction; trial counts; labels
        fig = plt.figure(figsize=(8.27, 11.69))
        gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1], hspace=0.45, wspace=0.5, left=0.3, right=0.76)
        a = fig.add_subplot(gs[0, :])
        y = np.arange(len(types))[::-1]
        for yi, at in zip(y, types):
            x = R[R.analysis_type == at]
            tested = x.zeta_p.notna().sum()
            d = x[x.significant].direction.value_counts()
            pos = d.get('positive', 0) + d.get('auditory', 0)
            neg = d.get('negative', 0) + d.get('whisker', 0)
            c = _type_color(at)
            a.barh(yi, pos / max(n_units, 1), color=c, lw=0)
            a.barh(yi, -neg / max(n_units, 1), color=c, alpha=0.45, lw=0)
            n1 = x.n_events_1.iloc[0] if len(x) else np.nan
            n2 = x.n_events_2.iloc[0] if len(x) else np.nan
            ev = f'{int(n1)}' if np.isnan(n2) else f'{int(n1)} / {int(n2)}'
            a.text(1.02, yi, f'{tested} tested; events {ev}', transform=a.get_yaxis_transform(), va='center',
                   fontsize=5.5)
        a.axvline(0, color='k', lw=0.5)
        a.set_yticks(y)
        a.set_yticklabels(types, fontsize=6)
        a.set_xlabel('Fraction of neurons with ZETA p < %.2f  (left: negative / whisker > auditory; '
                     'right: positive / auditory > whisker)' % ALPHA)
        a.set_title(f'{session_id}: {n_units} neurons', fontsize=9)
        lab = R.drop_duplicates('neuron_id')
        for j, col in enumerate(['modality_preference', 'sensory_label']):
            a = fig.add_subplot(gs[1, j])
            v = lab[col].value_counts()
            a.bar(range(len(v)), v.values, color='0.5', lw=0)
            a.set_xticks(range(len(v)))
            a.set_xticklabels(v.index, rotation=30, ha='right')
            a.set_ylabel('Neurons')
            a.set_title(f'{col} (from ZETA significance)')
        a = fig.add_subplot(gs[2, :])
        dec = lab[[c for c in lab.columns if c.endswith('decision')]].sum()
        a.bar(range(len(dec)), dec.values, color='0.5', lw=0)
        a.set_xticks(range(len(dec)))
        a.set_xticklabels(dec.index, rotation=20, ha='right')
        a.set_ylabel('Neurons')
        a.set_title('Decision labels (ROC definitions, ZETA significance)')
        pdf.savefig(fig)
        plt.close(fig)

        # page 2: latencies of significant responses (one-condition types)
        fig, axs = plt.subplots(len(one), 4, figsize=(8.27, 1.25 * len(one) + 0.8), squeeze=False)
        for i, at in enumerate(one):
            x = R[(R.analysis_type == at) & R.significant]
            c = _type_color(at)
            lo, hi = (LICK_WINDOW if at == 'spontaneous_licks' else (LATENZY_STIM_DUR[0], STIM_WINDOW_ONE[1]))
            b = np.arange(1000 * lo, 1000 * hi + 1, 5)
            for j, (col, lab_) in enumerate([('latency_s', 'latenZy onset'), ('ifr_peak_s', 'IFR peak')]):
                a = axs[i, j]
                for dr, al in (('positive', 1), ('negative', 0.4)):
                    v = 1000 * x[x.direction == dr][col].dropna()
                    a.hist(v, b, color=c, alpha=al, lw=0, label=f'{dr} ({len(v)})')
                a.set_xlabel(f'{lab_} (ms)')
                a.set_ylabel('Neurons')
                a.set_title(f'{at}: {lab_}', fontsize=6.5)
                a.legend(frameon=False, fontsize=5)
            a = axs[i, 2]
            for dr, mk in (('positive', 'o'), ('negative', 'v')):
                y_ = x[x.direction == dr]
                a.scatter(1000 * y_.latency_s, 1000 * y_.ifr_peak_s, s=3, color=c, lw=0,
                          alpha=0.6 if dr == 'positive' else 0.9, marker=mk)
            a.plot([1000 * lo, 1000 * hi], [1000 * lo, 1000 * hi], color='0.6', lw=0.5, ls=':')
            a.set_xlabel('latenZy onset (ms)')
            a.set_ylabel('IFR peak (ms)')
            a.set_title('positive (o) / negative (v)', fontsize=6.5)
            a = axs[i, 3]
            if 'neg_peak' in x and at != 'spontaneous_licks':
                v = x[x.neg_peak.astype(bool)]
                a.hist(1000 * v.neg_peak_s, np.arange(0, 1000 * NEG_PEAK_MAX_S + 1, 5), color=c, alpha=0.6, lw=0)
                a.set_title(f'early dips: {len(v)} neurons '
                            f'({(v.direction == "positive").sum()} positive-dominant)', fontsize=6.5)
            else:
                a.set_title('early dips: stimuli only', fontsize=6.5)
            a.set_xlabel('dip time (ms)')
            a.set_ylabel('Neurons')
        fig.suptitle(f'{session_id}: latencies of ZETA-significant responses', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        pdf.savefig(fig)
        plt.close(fig)

        # page 3: per-area fraction significant (positive / negative) for the one-condition types
        A = R[R.analysis_type.isin(one) & R.zeta_p.notna()]
        areas = A.drop_duplicates('neuron_id').area.value_counts()
        areas = areas[areas >= 10].index.tolist()
        if areas:
            fig, axs = plt.subplots(1, len(one), figsize=(8.27, 0.16 * len(areas) + 1.6), sharey=True, squeeze=False)
            for j, at in enumerate(one):
                a = axs[0, j]
                x = A[A.analysis_type == at]
                g = (x.assign(pos=(x.direction == 'positive').astype(float), neg=(x.direction == 'negative').astype(float))
                     .groupby('area')[['pos', 'neg']].mean().reindex(areas))
                yy = np.arange(len(areas))[::-1]
                a.barh(yy, g.pos, color=_type_color(at), lw=0)
                a.barh(yy, -g.neg, color=_type_color(at), alpha=0.45, lw=0)
                a.axvline(0, color='k', lw=0.4)
                a.set_title(at.replace('_', '\n', 1), fontsize=6)
                a.tick_params(labelsize=5.5)
                if j == 0:
                    a.set_yticks(yy)
                    a.set_yticklabels([f'{ar} ({areas_n})' for ar, areas_n in
                                       zip(areas, A.drop_duplicates('neuron_id').area.value_counts()[areas])])
            fig.supxlabel('Fraction significant (right positive, left negative)', fontsize=7)
            fig.suptitle(f'{session_id}: ZETA-responsive fraction per area (ccf parent; areas with >= 10 neurons)',
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

        # page 4: IFR heatmaps of positive (top block) and negative (bottom block) responses, each sorted by latenZy
        #         onset (then IFR peak), population means on top
        shown = [at for at in one if f'{at}__ifr' in ifr]
        if shown:
            fig, axs = plt.subplots(2, len(shown), figsize=(8.27, 7), squeeze=False,
                                    gridspec_kw=dict(height_ratios=[1, 3], hspace=0.25))
            for j, at in enumerate(shown):
                nid, M, t = ifr[f'{at}__neuron_id'], ifr[f'{at}__ifr'], 1000 * ifr[f'{at}__t_s']
                idx = {n: i for i, n in enumerate(nid)}
                x = R[(R.analysis_type == at) & R.significant].set_index('neuron_id')
                x = x[x.index.isin(idx)]
                c = _type_color(at)
                a = axs[1, j]
                blocks = [x[x.direction == dr].sort_values(['latency_s', 'ifr_peak_s']) for dr in ('positive', 'negative')]
                Zs = []
                for dr, xb, ls in zip(('positive', 'negative'), blocks, ('-', '--')):
                    if len(xb):
                        Z = M[[idx[n] for n in xb.index]].astype(float)
                        bw = (t >= 1000 * IFR_BASE_WINDOW[0]) & (t < 1000 * IFR_BASE_WINDOW[1])
                        Z = Z - (np.nanmean(Z[:, bw], 1, keepdims=True) if bw.sum() > 10
                                 else xb.mean_rate_1.to_numpy()[:, None])
                        Z = Z / np.nanmax(np.abs(Z), axis=1, keepdims=True)
                        Zs.append(Z)
                        axs[0, j].plot(t, np.nanmean(Z, 0), color=c, lw=0.8, ls=ls, label=f'{dr} ({len(xb)})')
                if Zs:
                    Z = np.vstack(Zs)
                    a.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='none',
                             extent=(t[0], t[-1], len(Z), 0), rasterized=True)
                    lat = 1000 * pd.concat(blocks).latency_s.to_numpy()
                    a.plot(lat, np.arange(len(Z)) + 0.5, '|', color='k', ms=1.5, mew=0.5)
                    a.axhline(len(blocks[0]), color='k', lw=0.6)
                    axs[0, j].legend(frameon=False, fontsize=4.5, loc='upper left')
                a.axvline(0, color='0.3', lw=0.4, ls='--')
                a.set_xlabel('Time from event (ms)', fontsize=6)
                a.set_title(f'{len(blocks[0])} pos / {len(blocks[1])} neg', fontsize=5.5)
                if j == 0:
                    a.set_ylabel('Neurons (positive above the line, negative below)', fontsize=6)
                a.tick_params(labelsize=5)
                axs[0, j].set_title(at.replace('_', '\n', 1), fontsize=6)
                axs[0, j].tick_params(labelsize=5)
                axs[0, j].axhline(0, color='0.6', lw=0.4, ls=':')
            axs[0, 0].set_ylabel('Mean norm. IFR', fontsize=6)
            fig.suptitle(f'{session_id}: IFR minus its pre-event mean, / max |.|, ZETA-significant neurons; '
                         f'sorted by latenZy onset (black ticks)', fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)

        # page 5: example neurons (strongest positive ZETA response per one-condition type)
        ex = []
        for at in shown:
            for dr in ('positive', 'negative'):
                x = R[(R.analysis_type == at) & (R.direction == dr)]
                if len(x):
                    ex.append((at, x.sort_values('zeta_z').iloc[-1]))
        if ex:
            byid = {u['neuron_id']: u for u in units}
            fig, axs = plt.subplots(len(ex), 2, figsize=(8.27, max(1.1 * len(ex) + 0.6, 4)), squeeze=False,
                                    gridspec_kw=dict(hspace=1.0))
            for i, (at, r) in enumerate(ex):
                ev_name, ctx, al = ONE_CONDITION[at]
                ev = events[(ev_name, ctx)]
                st = byid[r.neuron_id]['spike_times']
                lo, hi = (-0.2, 0.2)
                a = axs[i, 0]
                for k, e in enumerate(ev):
                    s = st[(st >= e + lo) & (st < e + hi)] - e
                    a.plot(1000 * s, np.full(len(s), k), '|', ms=1, mew=0.3, color='k')
                a.set_xlim(1000 * lo, 1000 * hi)
                a.set_ylabel('Event')
                a.set_title(f'{at}, {r.direction}: neuron {r.neuron_id} ({r.area}), ZETA z = {r.zeta_z:.1f}',
                            fontsize=6)
                a = axs[i, 1]
                t = 1000 * ifr[f'{at}__t_s']
                a.plot(t, ifr[f'{at}__ifr'][list(ifr[f'{at}__neuron_id']).index(r.neuron_id)], color=_type_color(at),
                       lw=0.8)
                a.axhline(r.mean_rate_1, color='0.5', lw=0.5, ls=':')
                for v, ls in ((r.latency_s, '--'), (r.ifr_peak_s, ':')):
                    if np.isfinite(v):
                        a.axvline(1000 * v, color='k', lw=0.6, ls=ls)
                a.set_xlim(1000 * lo, 1000 * hi)
                a.set_ylabel('IFR (Hz)')
                a.set_title(f'latenZy {1000 * r.latency_s:.0f} ms (--), IFR peak {1000 * r.ifr_peak_s:.0f} ms (:)',
                            fontsize=6.5)
            for a in axs[-1]:
                a.set_xlabel('Time from event (ms)')
            fig.suptitle(f'{session_id}: strongest positive and strongest negative ZETA response per event type',
                         fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
