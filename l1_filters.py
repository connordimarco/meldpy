"""
l1_filters.py
-------------
Signal-quality filters applied to the combined L1 time series.

Public entry point: despike()  — applies a centered 3-point median to B and plasma.

Shared utilities:
  - INTERP_LIMITS        — per-variable max-gap limits for interpolation.
  - interpolate_with_limits() — apply per-variable gap-limited interpolation.
"""
import numpy as np
import pandas as pd


# Max gap (in minutes) allowed when filling NaN runs per variable.
# Kept conservative so long outages remain NaN rather than being
# silently bridged with a ramp.
INTERP_LIMITS = {
    'Bx': 5,
    'By': 5,
    'Bz': 5,
    'Ux': 60,
    'Uy': 60,
    'Uz': 60,
    'rho': 60,
    'T': 60,
}


def interpolate_with_limits(df, limits=None, method='time'):
    """Interpolate each column with a per-variable max gap limit.

    Parameters
    ----------
    df : pd.DataFrame
    limits : dict[str, int] or None
        Maps column name -> max consecutive NaN steps to fill.
        Defaults to INTERP_LIMITS.
    method : str
        Interpolation method passed to pandas (default 'time').
    """
    if limits is None:
        limits = INTERP_LIMITS
    out = df.copy()
    for col, limit in limits.items():
        if col in out.columns:
            out[col] = out[col].interpolate(
                method=method,
                limit=limit,
                limit_area='inside',
            )
    return out


def median_filter_3(a, min_periods=2):
    """Apply a centered 3-point rolling median to a 1-D signal."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError('Input `a` must be one-dimensional.')
    if len(a) < 3:
        return a.copy()
    return pd.Series(a).rolling(
        window=3, center=True, min_periods=min_periods).median().to_numpy()


def despike(df):
    """Apply a centered 3-point median filter to B and plasma columns.

    Parameters
    ----------
    df : pd.DataFrame
        Combined L1 data with columns Bx/By/Bz/Ux/Uy/Uz/rho.

    Returns
    -------
    df_clean : pd.DataFrame
        Copy of *df* with the median filter applied.
    """
    print('  -> Running Despike (3-point median)...')
    df_clean = df.copy()
    for col in ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho']:
        if col in df_clean.columns:
            df_clean[col] = median_filter_3(df_clean[col].values)
    return df_clean
