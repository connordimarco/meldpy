"""
l1_filters.py
-------------
Signal-quality filters applied to the combined L1 time series.

Two core primitives:
  - limit_change()  — clamps point-to-point additive jumps.
  - limit_growth()  — clamps point-to-point multiplicative jumps.

Public entry point: despike()  — applies both to appropriate variables.
"""
import numpy as np


def limit_change(a, change, a2=None, change2=None):
    """Clamp consecutive differences to stay within allowed bounds.

    Parameters
    ----------
    a : array-like
        1-D input signal (may contain NaN).
    change : scalar or 2-element sequence
        Allowed change per step.  Scalar → symmetric (±|change|).
        Two-element → [lower, upper].
    a2 : array-like, optional
        Secondary signal.  When *a2[i] < a2[i-1] × 1.01* (not rising),
        the stricter *change2* limits are used for step *i*.
    change2 : scalar or 2-element sequence, optional
        Stricter limits used when a2 is not increasing.  Required when
        a2 is provided.

    Returns
    -------
    b : np.ndarray
        Filtered copy of *a*.
    """
    # Work with ndarray math no matter what comes in.
    a = np.asarray(a)

    if a.ndim != 1 or len(a) < 2:
        raise ValueError(
            'Input `a` must be a one-dimensional array with at least two elements.')

    if np.isscalar(change):
        changes = np.array([-abs(change), abs(change)])
    elif len(change) == 2:
        changes = np.array([min(change), max(change)])
    else:
        raise ValueError('`change` should be a scalar or a 2-element array.')

    if a2 is not None:
        # Optional second signal lets us relax/tighten limits conditionally.
        a2 = np.asarray(a2)
        if a2.shape != a.shape:
            raise ValueError('`a2` must have the same shape as `a`.')

        if np.isscalar(change2):
            changes2 = np.array([-abs(change2), abs(change2)])
        elif len(change2) == 2:
            changes2 = np.array([min(change2), max(change2)])
        else:
            raise ValueError(
                '`change2` should be a scalar or a 2-element array.')

    # First pass: clamp point-to-point deltas against `change`.
    b = a.copy()
    last_valid = np.nan
    for i in range(len(a)):
        if np.isnan(b[i]):
            continue
        if not np.isnan(last_valid):
            b[i] = np.clip(b[i], last_valid + changes[0],
                           last_valid + changes[1])
        last_valid = b[i]

    if a2 is not None:
        # Second pass: if a2 is not increasing, apply stricter limits.
        last_valid_b = np.nan
        last_valid_a2 = np.nan
        for i in range(len(a)):
            if np.isnan(a2[i]) or np.isnan(b[i]):
                if not np.isnan(a2[i]):
                    last_valid_a2 = a2[i]
                continue
            if not np.isnan(last_valid_b) and not np.isnan(last_valid_a2):
                if a2[i] < last_valid_a2 * 1.01:
                    b[i] = np.clip(b[i], last_valid_b +
                                   changes2[0], last_valid_b + changes2[1])
            last_valid_b = b[i]
            last_valid_a2 = a2[i]

    return b


def limit_growth(a, factor):
    """Clamp consecutive multiplicative jumps to within *factor*.

    Both positive and negative runs are handled correctly (negative values
    are bounded symmetrically).

    Parameters
    ----------
    a : array-like
        1-D input signal (may contain NaN).
    factor : float > 1
        Maximum allowed ratio between consecutive non-NaN values.

    Returns
    -------
    b : np.ndarray
        Filtered copy of *a*.
    """
    # Work with ndarray math no matter what comes in.
    a = np.asarray(a)

    if a.ndim != 1 or len(a) < 2:
        raise ValueError(
            'Input `a` must be a one-dimensional array with at least two elements.')

    if factor <= 1:
        raise ValueError('`factor` must be greater than 1.')

    # Clamp multiplicative jumps while preserving sign behavior.
    b = a.copy()
    last_valid = np.nan
    for i in range(len(a)):
        if np.isnan(b[i]):
            continue
        if np.isnan(last_valid):
            last_valid = b[i]
            continue
        if last_valid > 0 and b[i] > 0:
            b[i] = np.clip(b[i], last_valid / factor, last_valid * factor)
        elif last_valid < 0 and b[i] < 0:
            b[i] = np.clip(b[i], last_valid * factor, last_valid / factor)
        last_valid = b[i]

    return b


def despike(df):
    """Apply growth and change limiters to the combined L1 DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Combined L1 data with columns Bx/By/Bz/Ux/Uy/Uz/rho/T.

    Returns
    -------
    df_clean : pd.DataFrame
        Copy of *df* with density, temperature, and Ux filtered.
    """
    print('  -> Running Despike (Growth & Change limits)...')

    # Keep original input untouched.
    df_clean = df.copy()

    # Density and temperature get multiplicative growth checks.
    if 'rho' in df_clean.columns:
        df_clean['rho'] = limit_growth(df_clean['rho'].values, 1.3)

    if 'T' in df_clean.columns:
        df_clean['T'] = limit_growth(df_clean['T'].values, 1.3)

    # Velocity gets additive step limits with density-based relaxation.
    if 'Ux' in df_clean.columns and 'rho' in df_clean.columns:
        df_clean['Ux'] = limit_change(
            df_clean['Ux'].values,
            [-50, 30],
            df_clean['rho'].values,
            [-10, 20],
        )

    return df_clean
