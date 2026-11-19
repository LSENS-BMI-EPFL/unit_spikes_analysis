


import json
import pickle
import re
import warnings
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

"""
Minimal use
imec_id = _get_imec_id(unit["electrode_group"])
mp = build_motion_path(config, mouse_id, session_id, imec_id)
mo = load_motion(mp)
displacement = get_motion_at_times_and_depth(mo, times_s, unit_depth_um)


"""


def _get_imec_id(electrode_group) -> Optional[int]:
    if electrode_group is None:
        return None
    if isinstance(electrode_group, dict):
        val = electrode_group.get("imecID", electrode_group.get("imec_id"))
        return int(val) if val is not None else None
    if isinstance(electrode_group, str):
        try:
            d = json.loads(electrode_group.replace("'", '"'))
            if isinstance(d, dict):
                return int(d.get("imecID", d.get("imec_id")))
        except (json.JSONDecodeError, ValueError):
            pass
        m = re.search(r"imec(\d+)", electrode_group, re.IGNORECASE)
        if m:
            return int(m.group(1))
    try:
        return int(electrode_group)
    except (TypeError, ValueError):
        return None




def _find_session_data_root(data_root: str, mouse_id: str, session_id) -> Optional[Path]:
    mouse_dir = Path(data_root) / mouse_id
    if not mouse_dir.exists():
        return None
    candidates = sorted(p for p in mouse_dir.iterdir()
                        if p.is_dir() and p.name.startswith(mouse_id))
    if not candidates:
        return None
    sid_str  = str(session_id)
    matches  = [p for p in candidates if sid_str in p.name]
    if len(matches) == 1:
        return matches[0]
    if len(candidates) == 1:
        return candidates[0]
    warnings.warn(f"Ambiguous session folder for {mouse_id}/{session_id}; using most recent.")
    return candidates[-1]


def build_motion_path(config, mouse_id, session_id, imec_id) -> Path:
    session_root = _find_session_data_root(config["data_root"], mouse_id, session_id)
    if session_root is None:
        raise FileNotFoundError(f"No session folder: {mouse_id}/{session_id}")
    glob_pattern = "/".join(
        p.format(mouse_id=mouse_id, imec_id=imec_id, gate="g*")
        for p in config["motion_subpath_parts"]
    )
    matches = sorted(session_root.glob(glob_pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 motion match, found {len(matches)}: {matches}")
    return matches[0]


_MOTION_CACHE: dict = {}


def load_motion(motion_path: Path):
    key = str(motion_path)
    if key not in _MOTION_CACHE:
        import spikeinterface.core as sc
        _MOTION_CACHE[key] = sc.load(str(motion_path))
    return _MOTION_CACHE[key]


def get_motion_at_times_and_depth(motion, times_s, depth_um) -> np.ndarray:
    times_s = np.asarray(times_s, dtype=float)
    depths  = np.full_like(times_s, float(depth_um))
    if hasattr(motion, "get_displacement_at_time_and_depth"):
        return np.asarray(
            motion.get_displacement_at_time_and_depth(times_s, depths, segment_index=0),
            dtype=float).ravel()
    warnings.warn("Motion fallback to manual nearest-bin lookup.")
    temporal_bins = np.asarray(motion.temporal_bins_s[0])
    spatial_bins  = np.asarray(motion.spatial_bins_um)
    displacement  = np.asarray(motion.displacement[0])
    t_idx = np.searchsorted(temporal_bins, times_s)
    t_idx = np.clip(t_idx, 0, len(temporal_bins) - 1)
    left  = np.clip(t_idx - 1, 0, len(temporal_bins) - 1)
    t_idx = np.where(np.abs(temporal_bins[left] - times_s) <
                     np.abs(temporal_bins[t_idx] - times_s), left, t_idx)
    s_idx = int(np.argmin(np.abs(spatial_bins - depth_um)))
    return displacement[t_idx, s_idx]