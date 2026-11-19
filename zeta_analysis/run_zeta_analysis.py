#! /usr/bin/env/python3
"""
@project: unit_spikes_analysis
@file: zeta_analysis/run_zeta_analysis.py
@time: 2026-11-18

Run compute_unit_zeta on every session that has ROC results (<mouse>/<behaviour>_<day>/roc_analysis/
<mouse>_roc_results_new.csv; NWB path from its <mouse>_roc_config.json) and write to <mouse>/<behaviour>_<day>/zeta/.
Sessions with an existing <mouse>_zeta_results.csv are skipped unless --overwrite.
Equivalent to unit_spikes_analysis.py --analyses-single zeta, without loading the unit / trial tables.

Run (haas, repo root): PYTHONPATH=~/code/NWB_reader:. .venv/bin/python zeta_analysis/run_zeta_analysis.py
                       [--n-workers 100] [--mice AB116 MH069] [--overwrite] [--summary-only]
"""
import argparse
import glob
import json
import os
import time
import traceback

from zeta_analysis.zeta_utils import compute_unit_zeta, make_zeta_summary

RESULTS = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4'


def sessions(mice=None):
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, '*', '*', 'roc_analysis', '*_roc_results_new.csv'))):
        day_dir = os.path.dirname(os.path.dirname(f))
        mouse = os.path.basename(os.path.dirname(day_dir))
        if mice and mouse not in mice:
            continue
        cfg = os.path.join(os.path.dirname(f), f'{mouse}_roc_config.json')
        if not os.path.exists(cfg):
            print('no ROC config, skipped:', f)
            continue
        out.append((json.load(open(cfg))['source_nwb_file'], os.path.join(day_dir, 'zeta'), mouse))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n-workers', type=int, default=100)
    p.add_argument('--mice', nargs='+', default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--summary-only', action='store_true')
    a = p.parse_args()
    todo = sessions(a.mice)
    print(f'{len(todo)} sessions')
    t0 = time.time()
    for k, (nwb, out, mouse) in enumerate(todo):
        done = os.path.exists(os.path.join(out, f'{mouse}_zeta_results.csv'))
        try:
            if a.summary_only:
                if done:
                    make_zeta_summary(nwb, out)
            elif done and not a.overwrite:
                print('exists, skipped:', out)
                continue
            else:
                compute_unit_zeta(nwb, out, n_workers=a.n_workers)
        except Exception:
            print(f'[{nwb}] failed:\n{traceback.format_exc()}')
        print(f'{time.time() - t0:7.0f}s {k + 1}/{len(todo)} {os.path.basename(nwb)}', flush=True)
