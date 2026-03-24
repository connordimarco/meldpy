"""
l1_combine_T.py
---------------
Temperature combiner for the L1 pipeline.

T is handled separately from B and plasma because:
  - It spans orders of magnitude, making threshold-based quality checks unreliable.
  - Real propagation delays between spacecraft look identical to sensor disagreement.
  - ACE SWE proton T, WIND thermal-speed-derived T, and DSCOVR PLASMAG T have
    different calibrations that routinely disagree by 2-3x.

Strategy (two steps):
  1. Per minute: median of all available satellite T values.
     The median naturally rejects one outlier satellite without any explicit quality gate.
  2. Centered 3-point rolling median on the combined result to remove minute-level spikes.

No quality gating. No growth limits. Explainable in two sentences.
"""
import numpy as np
import pandas as pd

from l1_filters import median_filter_3


def combine_temperature(data_map, master_grid):
    """Merge proton temperature across ACE, DSCOVR, and WIND.

    Parameters
    ----------
    data_map : dict[str, pd.DataFrame]
        Per-satellite DataFrames keyed by 'ace', 'dscovr', 'wind'.
        Each must contain a 'T' column.
    master_grid : pd.DatetimeIndex
        Target 1-minute time grid.

    Returns
    -------
    pd.Series
        Combined temperature on master_grid.
    """
    sat_T = {}
    for sat in ('ace', 'dscovr', 'wind'):
        if sat in data_map and 'T' in data_map[sat].columns:
            s = data_map[sat]['T'].reindex(master_grid)
            # Fill isolated 1-2 minute dropouts before combining.
            s = s.interpolate(method='time', limit=2, limit_area='inside')
            sat_T[sat] = s
        else:
            sat_T[sat] = pd.Series(np.nan, index=master_grid)

    # Per-minute median: robust to one outlier satellite.
    combined = pd.DataFrame(sat_T).median(axis=1, skipna=True)

    # 3-point rolling median: removes single-minute spikes.
    combined = pd.Series(
        median_filter_3(combined.values),
        index=master_grid,
    )

    return combined
