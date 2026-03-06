"""
l1_quality.py
-------------
Plasma quality assessment for all three L1 satellites (ACE, DSCOVR, WIND).

Produces per-satellite, per-variable, per-timestep boolean bad-masks.
Five independent checks are applied:

  1. Flat-plateau detection   — stuck/near-constant instrument readings (DSCOVR)
  2. Outlier detection        — flags odd-one-out when 2-of-3 satellites agree
  3. Pairwise DSCOVR check    — catches bad DSCOVR in 2-satellite scenarios
  4. Physical-range check     — values outside physically plausible bounds
  5. NaN-fraction check       — high missing-data rate in rolling window
  6. Near-zero clearing       — Uy/Uz pinned near zero (DSCOVR only)

Main entry point: ``score_all_plasma()``.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Flat-plateau detection (generalised from the original Uy/Uz-only check)
# ---------------------------------------------------------------------------

# Per-variable rolling-window parameters.  Larger windows and looser
# thresholds for Ux (advisor: "Ux is often OK").
_PLATEAU_PARAMS = {
    'Ux':  {'window': 15, 'std_thresh': 1.0,    'max_unique': 3},
    'Uy':  {'window': 11, 'std_thresh': 0.08,   'max_unique': 3},
    'Uz':  {'window': 11, 'std_thresh': 0.08,   'max_unique': 3},
    'rho': {'window': 15, 'std_thresh': 0.05,   'max_unique': 3},
    'T':   {'window': 15, 'std_thresh': 500.0,  'max_unique': 3},
}


def _detect_flat_plateau(series, window, std_thresh, max_unique):
    """Return boolean Series — True where the signal looks stuck/flat."""
    s = series.copy()
    if s.empty or s.isna().all():
        return pd.Series(False, index=s.index)

    rolling_std = s.rolling(window=window, center=True, min_periods=5).std()
    rolling_unique = s.rolling(window=window, center=True, min_periods=5).apply(
        lambda x: pd.Series(x).dropna().nunique(), raw=False,
    )

    mask = (rolling_std <= std_thresh) & (rolling_unique <= max_unique)
    return mask.fillna(False)


def check_flat_plateau(df_dsc, variables=None):
    """Run flat-plateau detection on each requested DSCOVR plasma variable.

    Returns dict[str, pd.Series] mapping variable name -> bad mask (True=bad).
    """
    if variables is None:
        variables = list(_PLATEAU_PARAMS.keys())

    masks = {}
    for var in variables:
        if var not in _PLATEAU_PARAMS:
            continue
        if var not in df_dsc.columns or df_dsc[var].isna().all():
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue
        p = _PLATEAU_PARAMS[var]
        masks[var] = _detect_flat_plateau(
            df_dsc[var], p['window'], p['std_thresh'], p['max_unique'],
        )
    return masks


# ---------------------------------------------------------------------------
# 2. Inter-spacecraft consistency check
# ---------------------------------------------------------------------------

# Maximum acceptable deviation between DSCOVR and the reference spacecraft
# (ACE or WIND) over a rolling window.  Additive for velocity, multiplicative
# (ratio) for density and temperature.
_INTERSC_PARAMS = {
    'Ux':  {'mode': 'abs', 'threshold': 50.0,  'window': 31},
    'Uy':  {'mode': 'abs', 'threshold': 30.0,  'window': 31},
    'Uz':  {'mode': 'abs', 'threshold': 30.0,  'window': 31},
    'rho': {'mode': 'ratio', 'threshold': 3.0,  'window': 61},
    'T':   {'mode': 'ratio', 'threshold': 3.0,  'window': 31},
}


def _build_reference(df_ace, df_wind, var):
    """Pick a reference series for comparison: prefer WIND, fall back to ACE."""
    has_wind = (var in df_wind.columns) and not df_wind[var].isna().all()
    has_ace = (var in df_ace.columns) and not df_ace[var].isna().all()

    if has_wind and has_ace:
        # Average where both available; fill from whichever has coverage.
        ref = df_wind[var].combine_first(df_ace[var])
    elif has_wind:
        ref = df_wind[var]
    elif has_ace:
        ref = df_ace[var]
    else:
        return None
    return ref


def check_intersc_consistency(df_dsc, df_ace, df_wind, variables=None):
    """Flag DSCOVR minutes where it deviates heavily from ACE/WIND.

    Returns dict[str, pd.Series] mapping variable name -> bad mask (True=bad).
    """
    if variables is None:
        variables = list(_INTERSC_PARAMS.keys())

    masks = {}
    for var in variables:
        if var not in _INTERSC_PARAMS:
            continue
        if var not in df_dsc.columns or df_dsc[var].isna().all():
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue

        ref = _build_reference(df_ace, df_wind, var)
        if ref is None:
            # No reference available — cannot judge DSCOVR quality.
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue

        # Align to DSCOVR index.
        ref = ref.reindex(df_dsc.index)
        p = _INTERSC_PARAMS[var]
        w = p['window']

        if p['mode'] == 'abs':
            deviation = (df_dsc[var] - ref).abs()
            rolling_dev = deviation.rolling(
                window=w, center=True, min_periods=5,
            ).median()
            bad = rolling_dev > p['threshold']
        else:
            # Ratio mode for density/temperature.
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = df_dsc[var] / ref
            # Flag where ratio departs from 1 by more than the threshold factor.
            log_ratio = np.log(ratio.clip(lower=1e-10)).abs()
            log_thresh = np.log(p['threshold'])
            rolling_lr = log_ratio.rolling(
                window=w, center=True, min_periods=5,
            ).median()
            bad = rolling_lr > log_thresh

        masks[var] = bad.fillna(False)

    return masks


def check_outlier_satellite(df_ace, df_dsc, df_wind, variables=None):
    """Flag the outlier when 2 of 3 satellites agree and 1 disagrees.

    A satellite is flagged only when the other two *agree* with each other
    (pairwise deviation within threshold) **and** it disagrees with both.
    This avoids cross-contamination that occurs when using a combined
    reference from two satellites that may include a bad one.

    Returns
    -------
    result : dict[int, dict[str, pd.Series]]
        ``result[sat_code][var]`` is True where that satellite is an outlier.
    """
    if variables is None:
        variables = list(_INTERSC_PARAMS.keys())

    idx = df_ace.index
    sat_series_all = {1: df_ace, 2: df_dsc, 3: df_wind}
    codes = [1, 2, 3]
    result = {c: {} for c in codes}

    for var in variables:
        p = _INTERSC_PARAMS[var]
        w = p['window']

        s = {}
        for c, df in sat_series_all.items():
            s[c] = df[var].reindex(idx) if var in df.columns else pd.Series(np.nan, index=idx)

        # Pairwise agreement masks for every combination.
        pair_ok = {}
        for a, b in [(1, 2), (1, 3), (2, 3)]:
            if p['mode'] == 'abs':
                dev = (s[a] - s[b]).abs()
                roll = dev.rolling(window=w, center=True, min_periods=5).median()
                pair_ok[(a, b)] = (roll <= p['threshold']).fillna(True)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = s[a] / s[b]
                log_r = np.log(ratio.clip(lower=1e-10)).abs()
                log_thresh = np.log(p['threshold'])
                roll = log_r.rolling(window=w, center=True, min_periods=5).median()
                pair_ok[(a, b)] = (roll <= log_thresh).fillna(True)

        for c in codes:
            others = sorted(x for x in codes if x != c)
            a, b = others
            others_agree = pair_ok[(min(a, b), max(a, b))]
            disagrees_a = ~pair_ok[(min(c, a), max(c, a))]
            disagrees_b = ~pair_ok[(min(c, b), max(c, b))]
            result[c][var] = (others_agree & disagrees_a & disagrees_b).fillna(False)

    return result


# ---------------------------------------------------------------------------
# 3. Physical range checks
# ---------------------------------------------------------------------------

_PHYSICAL_BOUNDS = {
    'Ux':  {'min': -2500.0, 'max': -150.0},   # must flow Earthward
    'Uy':  {'min': -200.0,  'max': 200.0},
    'Uz':  {'min': -200.0,  'max': 200.0},
    'rho': {'min': 0.1,     'max': 100.0},
    'T':   {'min': 1000.0,  'max': 5.0e6},
}


def check_physical_range(df_dsc, variables=None):
    """Flag DSCOVR values outside physically plausible bounds.

    Returns dict[str, pd.Series] mapping variable name -> bad mask (True=bad).
    """
    if variables is None:
        variables = list(_PHYSICAL_BOUNDS.keys())

    masks = {}
    for var in variables:
        if var not in _PHYSICAL_BOUNDS:
            continue
        if var not in df_dsc.columns or df_dsc[var].isna().all():
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue
        bounds = _PHYSICAL_BOUNDS[var]
        bad = (df_dsc[var] < bounds['min']) | (df_dsc[var] > bounds['max'])
        masks[var] = bad.fillna(False)
    return masks


# ---------------------------------------------------------------------------
# 4. NaN-fraction (data-gap) metric
# ---------------------------------------------------------------------------

def check_nan_fraction(df_dsc, variables=None, window=60, threshold=0.5):
    """Flag windows where DSCOVR has > threshold fraction of NaN.

    A high NaN rate suggests the instrument is struggling; surrounding
    non-NaN points in the same window are also suspect.

    Returns dict[str, pd.Series] mapping variable name -> bad mask (True=bad).
    """
    if variables is None:
        variables = ['Ux', 'Uy', 'Uz', 'rho', 'T']

    masks = {}
    for var in variables:
        if var not in df_dsc.columns:
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue
        nan_flag = df_dsc[var].isna().astype(float)
        frac = nan_flag.rolling(
            window=window, center=True, min_periods=10).mean()
        masks[var] = (frac > threshold).fillna(False)
    return masks


# ---------------------------------------------------------------------------
# 5. Near-zero Uy/Uz (legacy check, moved here from l1_combine)
# ---------------------------------------------------------------------------

def check_near_zero(df_dsc, variables=None, atol=0.5):
    """Flag Uy/Uz near-zero values as suspect.

    Returns dict[str, pd.Series] mapping variable name -> bad mask (True=bad).
    """
    if variables is None:
        variables = ['Uy', 'Uz']

    masks = {}
    for var in variables:
        if var not in df_dsc.columns:
            masks[var] = pd.Series(False, index=df_dsc.index)
            continue
        masks[var] = df_dsc[var].abs().le(atol) & df_dsc[var].notna()
    return masks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

PLASMA_VARS = ['Ux', 'Uy', 'Uz', 'rho', 'T']

def score_all_plasma(df_ace, df_dsc, df_wind):
    """Evaluate plasma quality for every satellite, variable, and minute.

    Uses two complementary inter-spacecraft strategies:

    * **Outlier detection** (all satellites) — flags the odd-one-out when
      the other two agree with each other.  Requires 3 satellites.
    * **Pairwise DSCOVR check** — the original ``check_intersc_consistency``
      run on DSCOVR against ACE/WIND.  Catches DSCOVR issues when only 2
      satellites have data (e.g. during ACE outages).

    Physical-range, NaN-fraction, flat-plateau (DSCOVR), and near-zero
    (DSCOVR) checks are applied on top.

    Returns
    -------
    all_bad : dict[int, dict[str, pd.Series]]
        ``all_bad[sat_code][var]`` is True where that satellite’s value
        should not be used.  Codes: 1 = ACE, 2 = DSCOVR, 3 = WIND.
    """
    idx = df_ace.index
    df_ace = df_ace.reindex(idx) if not df_ace.empty else pd.DataFrame(index=idx)
    df_dsc = df_dsc.reindex(idx) if not df_dsc.empty else pd.DataFrame(index=idx)
    df_wind = df_wind.reindex(idx) if not df_wind.empty else pd.DataFrame(index=idx)

    sat_dfs = {1: df_ace, 2: df_dsc, 3: df_wind}
    sat_names = {1: 'ACE', 2: 'DSCOVR', 3: 'WIND'}

    # --- Outlier detection (works when 3 satellites are present) ---
    outlier_masks = check_outlier_satellite(df_ace, df_dsc, df_wind, PLASMA_VARS)

    # --- DSCOVR pairwise check (works with 2+ satellites) ---
    dsc_intersc = check_intersc_consistency(df_dsc, df_ace, df_wind, PLASMA_VARS)

    all_bad = {}
    for code, df_target in sat_dfs.items():
        range_m = check_physical_range(df_target, PLASMA_VARS)
        nan_m = check_nan_fraction(df_target, PLASMA_VARS)

        # DSCOVR-specific instrument checks.
        plateau_m = check_flat_plateau(df_target, PLASMA_VARS) if code == 2 else {}
        zero_m = check_near_zero(df_target, ['Uy', 'Uz']) if code == 2 else {}

        bad = {}
        for var in PLASMA_VARS:
            composite = pd.Series(False, index=idx)
            # Outlier check (any satellite).
            if var in outlier_masks.get(code, {}):
                composite = composite | outlier_masks[code][var]
            # DSCOVR pairwise check.
            if code == 2 and var in dsc_intersc:
                composite = composite | dsc_intersc[var]
            # Standard per-satellite checks.
            for masks in [range_m, nan_m, plateau_m, zero_m]:
                if var in masks:
                    composite = composite | masks[var]
            bad[var] = composite

        n_flagged = {v: int(m.sum()) for v, m in bad.items()}
        print(f'  {sat_names[code]} quality: flagged bad: {n_flagged}')
        all_bad[code] = bad

    return all_bad
