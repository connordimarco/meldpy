"""
l1_filters.py
-------------
Signal-quality filters applied to the combined L1 time series.

Public entry points:
  - despike()              — centered 3-point median on B and plasma.
  - smooth_transitions()   — boxcar smoothing at large source-change steps (plasma only).

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


# ---------------------------------------------------------------------------
# Transition smoothing
# ---------------------------------------------------------------------------

# Plasma columns and how their jump magnitude C is computed.
#   'pct' : C = 100 * (max(|M1|, |M2|) / min(|M1|, |M2|) - 1)  [%]
#   'abs' : C = |M2 - M1|                                        [km/s]
_TRANSITION_COLS = {
    'Ux': 'pct',  # treat as positive magnitude
    'Uy': 'abs',
    'Uz': 'abs',
    'rho': 'pct',
    'T': 'pct',
}
_CMAX_DEFAULT = 20.0   # threshold below which no smoothing is applied
_WMAX_DEFAULT = 60     # maximum smoothing window [min]
_RATE_DEFAULT = 5.0    # W = C / rate, clipped to WMAX


def _jump_magnitude(m1, m2, col_type):
    """Return jump magnitude C given consecutive values and column type."""
    if col_type == 'pct':
        a, b = abs(m1), abs(m2)
        if min(a, b) == 0:
            return 0.0
        return 100.0 * (max(a, b) / min(a, b) - 1.0)
    return abs(m2 - m1)


def _apply_boxcar(smoothed, original, i, w):
    """Write a centered rolling mean of width *w* into *smoothed* around index *i*."""
    n = len(original)
    half = w // 2
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    ext_lo = max(0, lo - half)
    ext_hi = min(n, hi + half)
    rolled = (pd.Series(original[ext_lo:ext_hi])
              .rolling(w, center=True, min_periods=1)
              .mean().values)
    smoothed[lo:hi] = rolled[(lo - ext_lo):(lo - ext_lo) + (hi - lo)]


def smooth_transitions(df, cmax=_CMAX_DEFAULT, wmax=_WMAX_DEFAULT,
                       rate=_RATE_DEFAULT):
    """Smooth large step changes in plasma output using boxcar averaging.

    For each plasma column finds minute-to-minute jumps where the magnitude
    C exceeds *cmax*.  Replaces the values in a window of width W around the
    step with a rolling mean of the same width computed from the *original*
    (pre-smoothing) values.  This turns a hard step into a gradual ramp.

        W = nint(min(wmax, C / rate))   [minutes]

    Parameters
    ----------
    df : pd.DataFrame
        Combined L1 data (must be on a 1-minute grid, no NaN gaps).
    cmax : float
        Minimum jump magnitude to trigger smoothing.
        Units: % for rho, T, |Ux|; km/s for Uy, Uz.
    wmax : int
        Maximum smoothing window in minutes.
    rate : float
        Rate parameter: W = C / rate.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with large plasma transitions smoothed.
    """
    out = df.copy()

    for col, col_type in _TRANSITION_COLS.items():
        if col not in out.columns:
            continue

        original = out[col].values.astype(float)
        smoothed = original.copy()

        for i in range(1, len(original)):
            m1, m2 = original[i - 1], original[i]
            if not (np.isfinite(m1) and np.isfinite(m2)):
                continue
            c = _jump_magnitude(m1, m2, col_type)
            if c <= cmax:
                continue
            w = max(2, int(np.round(min(wmax, c / rate))))
            _apply_boxcar(smoothed, original, i, w)

        out[col] = smoothed

    return out


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
