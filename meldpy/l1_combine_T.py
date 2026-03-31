"""
l1_combine_T.py
---------------
Temperature combiner for the L1 pipeline.

T is handled separately from B and plasma because:
  - It spans orders of magnitude, making threshold-based quality checks unreliable.
  - Real propagation delays between spacecraft look identical to sensor disagreement.
  - ACE SWE proton T, WIND thermal-speed-derived T, and DSCOVR PLASMAG T have
    different calibrations that routinely disagree by 2-3x.

Strategy (three steps):
  1. Per-satellite 3-point median to remove single-minute spikes.
  2. Per-satellite spikiness filter: if a satellite's T has high rolling log-std
     over an 11-minute window (> 0.5, equivalent to ~1.65x typical variation),
     those minutes are excluded from the combination.
  3. Per-minute geometric median across available satellites:
       exp(median(log(T_values)))
     Works correctly at any spread — no threshold, no source-switching logic.
     With 2 satellites this is the geometric mean (splits the difference in
     log-space). With 3 it returns the log-space middle value.
  4. Final 3-point rolling median on the combined result.
"""
import numpy as np
import pandas as pd

from .l1_filters import median_filter_3

# Rolling log-std threshold above which a satellite's T is considered too noisy
# to contribute to the combination. Evaluated over an 11-minute window.
# ACE and WIND typically stay well below 0.3; DSCOVR exceeds this during
# multi-minute oscillation episodes (e.g. May 22 2024: 37% of minutes flagged
# vs 0-3% on quiet days).
_T_SPIKY_LOG_STD = 0.5
_T_SPIKY_WINDOW = 11


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
            s = s.interpolate(method='time', limit=2, limit_area='inside')
            # Step 1: per-satellite 3-pt median removes single-minute spikes.
            s = pd.Series(median_filter_3(s.values), index=master_grid)
            # Step 2: exclude minutes where this satellite's T is too noisy.
            log_std = (
                np.log(s.clip(lower=1))
                .rolling(_T_SPIKY_WINDOW, center=True, min_periods=5)
                .std()
            )
            s = s.where(log_std <= _T_SPIKY_LOG_STD, other=np.nan)
            sat_T[sat] = s
        else:
            sat_T[sat] = pd.Series(np.nan, index=master_grid)

    # Step 3: geometric median — median in log-space, exponentiated back.
    # No threshold, no source-switching. Works correctly regardless of spread.
    df = pd.DataFrame(sat_T)
    log_df = np.log(df.clip(lower=1))
    log_median = log_df.median(axis=1, skipna=True)
    out = np.where(df.notna().any(axis=1), np.exp(log_median), np.nan)

    combined = pd.Series(out, index=master_grid)
    # Step 4: final 3-pt rolling median smooths minute-level residual noise.
    combined = pd.Series(median_filter_3(combined.values), index=master_grid)
    return combined
