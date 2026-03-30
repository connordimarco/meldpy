"""
l1_combine.py
-------------
Combines per-satellite L1 .dat files into a single merged output.

Core responsibilities:
  - Fills short intra-satellite gaps before merging.
  - Calls score_all_plasma() to obtain per-satellite quality bad-masks.
  - Selects the best available source for every variable/minute using
    agreement-first rules plus previous-value continuity fallback.
  - Writes L1_combined.dat (with nSat provenance) and
    ballistically propagated products (IMF_14Re.dat, IMF_32Re.dat).
  - Optionally uses prev_day / next_day context to warm up rolling filters
    across day boundaries before slicing today's data for output.

Public entry point: create_combined_l1_files()
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

from l1_combine_T import combine_temperature
from l1_filters import smooth_transitions, interpolate_with_limits, INTERP_LIMITS
from l1_propagation import ballistic_propagation
from l1_quality import score_all_plasma
from l1_readers import read_l1_data


SAT_CODE = {'ace': 1, 'dscovr': 2, 'wind': 3}


def _fill_short_gaps(df, max_gap=2):
    if df.empty:
        return df
    return df.interpolate(method='time', limit=max_gap, limit_area='inside')


def _switch_threshold(col):
    thresholds = {
        'Bx': 8.0,
        'By': 8.0,
        'Bz': 8.0,
        'Ux': 80.0,
        'Uy': 40.0,
        'Uz': 40.0,
        'rho': 2.0,
    }
    return thresholds.get(col, np.inf)


def _agree(v1, v2, col):
    """Return True when two values are close enough to be considered agreeing."""
    return abs(v1 - v2) <= _switch_threshold(col)


def _fallback_source(values, available_codes, prev_value):
    """Fallback source when no pair agrees.

    Startup (no previous value): prefer WIND when available.
    Steady-state: choose source closest to previous output value.
    """
    if np.isfinite(prev_value):
        return min(available_codes, key=lambda c: abs(values[c] - prev_value))
    if 3 in available_codes:
        return 3
    return available_codes[0]


def _select_column_with_continuity(col, sat_series, bad_masks=None):
    """Merge one variable across satellites with continuity and quality logic.

    Parameters
    ----------
    col : str
        Variable name.
    sat_series : dict[str, pd.Series]
        Per-satellite series for this variable.
    bad_masks : dict[int, dict[str, pd.Series]] or None
        Per-satellite, per-variable boolean masks from the quality scorer.
        ``bad_masks[sat_code][var]`` is True where that satellite is bad.
    """
    # Minimum consecutive minutes the alternative satellite must be closer
    # before the fallback source switches.  Prevents single-minute noise
    # from triggering a source change in the 2-sat no-agreement path.
    _SWITCH_MIN = 3

    index = sat_series['ace'].index
    out_vals = np.full(len(index), np.nan, dtype=float)
    out_nsat = np.zeros(len(index), dtype=int)
    out_source = [None] * len(index)   # frozenset of sat codes contributing each minute

    prev_value = np.nan
    locked_source = None   # satellite code currently committed in fallback path
    switch_count = 0       # consecutive minutes the preferred candidate != locked

    for i, _ in enumerate(index):
        values = {
            code: sat_series[sat].iloc[i]
            for sat, code in SAT_CODE.items()
            if pd.notna(sat_series[sat].iloc[i])
        }

        # --- Quality gate: drop ANY satellite flagged bad ---
        if bad_masks is not None:
            for code in list(values.keys()):
                sat_masks = bad_masks.get(code)
                if sat_masks is not None and col in sat_masks:
                    if bool(sat_masks[col].iloc[i]):
                        values.pop(code, None)

        available = sorted(values.keys())
        n_sat = len(available)
        out_nsat[i] = n_sat

        if n_sat == 0:
            continue

        if n_sat == 1:
            out_vals[i] = values[available[0]]
            out_source[i] = frozenset(available)
            prev_value = out_vals[i]
            continue

        # Agreement-first logic:
        #  - all 3 agree -> median
        #  - any agreeing pair -> mean of that pair
        #  - none agree -> fallback source (WIND at startup, else closest to prev)
        pairs = []
        for p_idx, c1 in enumerate(available):
            for c2 in available[p_idx + 1:]:
                if _agree(values[c1], values[c2], col):
                    pairs.append((c1, c2))

        if n_sat == 3 and len(pairs) == 3:
            out_vals[i] = np.median([values[1], values[2], values[3]])
            out_source[i] = frozenset([1, 2, 3])
            prev_value = out_vals[i]
            continue

        if pairs:
            best_pair = min(pairs, key=lambda p: abs(
                values[p[0]] - values[p[1]]))
            out_vals[i] = 0.5 * (values[best_pair[0]] + values[best_pair[1]])
            out_source[i] = frozenset(best_pair)
            prev_value = out_vals[i]
            continue

        # Fallback: no agreeing pair.  Use hysteresis to prevent oscillation
        # when two satellites alternate which is slightly closer to prev_value.
        candidate = _fallback_source(values, available, prev_value)

        if locked_source is None or locked_source not in available:
            # Cold-start or locked satellite disappeared — accept immediately.
            locked_source = candidate
            switch_count = 0
        elif candidate == locked_source:
            switch_count = 0
        else:
            switch_count += 1
            if switch_count >= _SWITCH_MIN:
                locked_source = candidate
                switch_count = 0

        out_vals[i] = values[locked_source]
        out_source[i] = frozenset([locked_source])
        prev_value = out_vals[i]

    return (pd.Series(out_vals, index=index),
            pd.Series(out_nsat, index=index),
            pd.Series(out_source, index=index))


def _read_sat_positions(pos_file):
    """Return per-satellite noon X positions in km from L1_satpos.dat.

    Returns dict with keys 'ace', 'dscovr', 'wind'. Missing values are NaN.
    """
    result = {'ace': np.nan, 'dscovr': np.nan, 'wind': np.nan}
    if not os.path.exists(pos_file):
        return result
    try:
        with open(pos_file, 'r', encoding='utf-8') as f:
            data_started = False
            for line in f:
                if line.strip().startswith('#START'):
                    data_started = True
                    continue
                if not data_started:
                    continue
                parts = line.split()
                if len(parts) >= 15:
                    result['ace']    = float(parts[6])  * 6371.0
                    result['dscovr'] = float(parts[9])  * 6371.0
                    result['wind']   = float(parts[12]) * 6371.0
                    break
    except Exception as e:
        print(f'  Warning: Could not read position file ({e}).')
    return result


def combine_data_priority(data_map, master_grid):
    # Align each satellite to the same timeline before merging.
    df_ace = _fill_short_gaps(data_map.get('ace', pd.DataFrame(
        index=master_grid)).reindex(master_grid))
    df_dsc = _fill_short_gaps(data_map.get('dscovr', pd.DataFrame(
        index=master_grid)).reindex(master_grid))
    df_win = _fill_short_gaps(data_map.get('wind', pd.DataFrame(
        index=master_grid)).reindex(master_grid))

    # --- Run quality scorer across all three satellites ---
    print('  Running plasma quality assessment for all satellites...')
    all_bad_masks = score_all_plasma(df_ace, df_dsc, df_win)

    # T is excluded here — handled separately by combine_temperature().
    cols = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho']
    plasma_cols = {'Ux', 'Uy', 'Uz', 'rho'}
    df_combined = pd.DataFrame(index=master_grid, columns=cols)
    nsat_map = {}
    source_map = {}   # col -> pd.Series of frozenset(sat_codes)

    # Process each physical variable with priority + continuity rules.
    for col in cols:
        sat_series = {
            'ace': df_ace[col] if col in df_ace else pd.Series(np.nan, index=master_grid),
            'dscovr': df_dsc[col] if col in df_dsc else pd.Series(np.nan, index=master_grid),
            'wind': df_win[col] if col in df_win else pd.Series(np.nan, index=master_grid),
        }

        # Quality masks for plasma variables; None for magnetic field.
        col_masks = all_bad_masks if col in plasma_cols else None

        values, n_sat, source = _select_column_with_continuity(
            col,
            sat_series,
            bad_masks=col_masks,
        )
        df_combined[col] = values
        nsat_map[col] = n_sat
        source_map[col] = source

    # Interpolate short NaN gaps left by quality-gating.
    df_combined = df_combined.interpolate(
        method='time', limit=30, limit_area='inside')

    provenance = pd.DataFrame(index=master_grid)
    provenance['nSat'] = nsat_map['Ux'].astype('Int64')

    return df_combined, provenance, source_map


def create_combined_l1_files(day, prev_day=None, next_day=None,
                             boundaries_re=(14, 32)):
    """Build combined L1 products for *day* using rolling neighbour context.

    When *prev_day* and/or *next_day* are supplied the quality-scoring,
    satellite-selection, and despiking algorithms run over the full
    multi-day window so that day-boundary artefacts (cold-start of rolling
    filters, etc.) are eliminated.  Only the portion corresponding to *day*
    is written to disk.
    """
    dt_start = datetime.strptime(day, '%Y-%m-%d')
    output_dir = dt_start.strftime('L1/%Y/%m/%d')
    os.makedirs(output_dir, exist_ok=True)
    satellites = ['ace', 'dscovr', 'wind']
    numeric_cols = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']

    # ---- Gather context-window days in chronological order ----
    context_days = []
    if prev_day:
        context_days.append(prev_day)
    context_days.append(day)
    if next_day:
        context_days.append(next_day)

    data_map = {}
    print(
        f"\nProcessing L1 data for {day}  (context window: {context_days})...")

    # Load per-satellite .dat files for every day in the window.
    for sat in satellites:
        frames = []
        for d in context_days:
            dt_d = datetime.strptime(d, '%Y-%m-%d')
            d_dir = dt_d.strftime('L1/%Y/%m/%d')
            fname = os.path.join(d_dir, f'L1_{sat}.dat')
            df = read_l1_data(fname)
            if not df.empty:
                frames.append(df[numeric_cols])
        if frames:
            combined = pd.concat(frames).sort_index()
            data_map[sat] = combined[~combined.index.duplicated(keep='first')]

    if not data_map:
        print('No satellite data found. Skipping.')
        return

    # ---- Gap-fill across day boundaries using the full context window ----
    # Stage 1 fills gaps within each day but can't bridge trailing/leading
    # edges where the adjacent day provides the missing bracket.  Now that
    # all three days are loaded together we re-apply the same per-variable
    # limits so those cross-midnight gaps are filled before any merging.
    for sat in list(data_map.keys()):
        data_map[sat] = interpolate_with_limits(data_map[sat], INTERP_LIMITS)

    # ---- Build a master grid spanning the full context window ----
    window_start = pd.Timestamp(context_days[0])
    window_end = pd.Timestamp(context_days[-1]) + pd.Timedelta(days=1)
    n_minutes = int((window_end - window_start).total_seconds() / 60)
    master_grid = pd.date_range(start=window_start, periods=n_minutes,
                                freq='1min')

    # ---- Propagate satellites to a common reference position ----
    # Find the satellite closest to Earth (smallest X) and shift all others
    # forward in time to that X before combining.  This ensures the combine
    # step compares measurements of the same solar-wind parcel.
    pos_file = os.path.join(output_dir, 'L1_satpos.dat')
    sat_x_km = _read_sat_positions(pos_file)

    available_x = {sat: sat_x_km[sat]
                   for sat in data_map
                   if np.isfinite(sat_x_km.get(sat, np.nan))}
    if available_x:
        ref_sat = min(available_x, key=lambda s: available_x[s])
        x_ref_km = available_x[ref_sat]
        pos_summary = ', '.join(
            f'{s.upper()}: {available_x[s] / 6371.0:.1f} Re'
            for s in ('ace', 'dscovr', 'wind') if s in available_x
        )
        print(f'  Reference position: {ref_sat.upper()} at '
              f'{x_ref_km / 6371.0:.1f} Re  ({pos_summary})')
        for sat, df_sat in list(data_map.items()):
            x_sat = available_x.get(sat, np.nan)
            if not np.isfinite(x_sat) or x_sat <= x_ref_km:
                continue
            df_renamed = df_sat.rename(
                columns={'Ux': 'Vx Velocity, km/s, GSE'})
            orbit = pd.Series({'X_GSE': x_sat})
            df_prop = ballistic_propagation(
                orbit, df_renamed, target_x_km=x_ref_km)
            data_map[sat] = df_prop.rename(
                columns={'Vx Velocity, km/s, GSE': 'Ux'})
    else:
        x_ref_km = 1.5e6
        print('  No satellite positions available; using default 1.5e6 km.')

    # Quality-score + satellite-select over the full window (B and plasma, not T).
    df_combined, provenance, source_map = combine_data_priority(data_map, master_grid)

    # Combine T separately: median of available satellites + 3-point median smooth.
    df_combined['T'] = combine_temperature(data_map, master_grid)

    # Smooth large step changes at source transitions (plasma only).
    # Applied on the full context window so the smoothing window has data
    # on both sides of transitions near midnight.
    # Build per-column boolean mask: True at minutes where the contributing
    # satellite set changed relative to the previous minute.
    source_changed = {}
    for col, src in source_map.items():
        vals = src.values
        changed = np.zeros(len(vals), dtype=bool)
        for k in range(1, len(vals)):
            if vals[k] is not None and vals[k - 1] is not None:
                changed[k] = vals[k] != vals[k - 1]
        source_changed[col] = pd.Series(changed, index=src.index)
    df_combined = smooth_transitions(df_combined, source_changed=source_changed)

    # ---- Slice out only *today* for file output ----
    today_start = pd.Timestamp(day)
    today_end = today_start + pd.Timedelta(days=1)
    today_mask = (df_combined.index >= today_start) & \
                 (df_combined.index < today_end)
    df_today = df_combined.loc[today_mask].copy()
    prov_today = provenance.loc[today_mask].copy()

    # Final pass: fill any remaining NaN gaps with linear interpolation.
    df_today = df_today.interpolate(method='linear')

    # Write unpropagated combined file.
    outfile_comb = os.path.join(output_dir, 'L1_combined.dat')
    with open(outfile_comb, 'w', encoding='utf-8') as f:
        f.write(
            f'Combined L1 Data for {day} (Unpropagated) (GSM nT, km/s, cm^-3, K)\n')
        f.write('year  mo  dy  hr  mn  sc msc Bx By Bz Ux Uy Uz rho T nSat\n')
        f.write('#START\n')
        for t, row in df_today.iterrows():
            if pd.isna(row['Bx']):
                continue
            n_sat_val = int(prov_today.at[t, 'nSat']) if pd.notna(
                prov_today.at[t, 'nSat']) else 0
            f.write(
                f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} {t.second:2d} {t.microsecond//1000:3d} "
                f"{row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} "
                f"{row['Ux']:9.2f} {row['Uy']:9.2f} {row['Uz']:9.2f} "
                f"{row['rho']:9.4f} {row['T']:10.1f} {n_sat_val:2d}\n"
            )
    print(f'Created {outfile_comb}')

    # x_ref_km was determined during the propagate-to-reference step above.
    mock_orbit = pd.Series({'X_GSE': x_ref_km})

    # Propagator expects Vx under this legacy column name.
    # Propagate only today's slice.
    df_prop_input = df_today.copy()
    df_prop_input = df_prop_input.rename(
        columns={'Ux': 'Vx Velocity, km/s, GSE'})

    # Write one propagated product per requested boundary.
    for b_re in boundaries_re:
        target_km = b_re * 6371.0
        print(f'  -> Propagating to {b_re} Re ({target_km:.0f} km)...')

        df_propagated = ballistic_propagation(
            mock_orbit, df_prop_input, target_x_km=target_km)
        df_propagated = df_propagated.rename(
            columns={'Vx Velocity, km/s, GSE': 'Ux'})

        filename = f'IMF_{b_re}Re.dat'
        outfile_prop = os.path.join(output_dir, filename)

        with open(outfile_prop, 'w', encoding='utf-8') as f:
            f.write(
                f'Propagated L1 Data for {day} (Target: {b_re} Re) (GSM nT, km/s, cm^-3, K)\n')
            f.write('year mo dy hr mn sc msc Bx By Bz Ux Uy Uz rho T\n')
            f.write('#START\n')
            for t, row in df_propagated.iterrows():
                if pd.isna(row['Bx']):
                    continue
                f.write(
                    f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} {t.second:2d} {t.microsecond//1000:3d} "
                    f"{row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} "
                    f"{row['Ux']:9.2f} {row['Uy']:9.2f} {row['Uz']:9.2f} "
                    f"{row['rho']:9.4f} {row['T']:10.1f}\n"
                )
        print(f'    Created {outfile_prop}')
