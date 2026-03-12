"""
l1_combine.py
-------------
Combines per-satellite L1 .dat files into a single merged output.

Core responsibilities:
  - Fills short intra-satellite gaps before merging.
  - Calls score_all_plasma() to obtain per-satellite quality bad-masks.
  - Selects the best available source for every variable/minute using a
    priority hierarchy: median (3-sat) > WIND > ACE/DSCOVR rules.
  - Applies a continuity guard to prevent large jumps when the source
    satellite changes.
  - Writes L1_combined.dat (with nSat and satUsed provenance columns) and
    ballistically propagated products (IMF_14Re.dat, IMF_32Re.dat).
  - Optionally uses prev_day / next_day context to warm up rolling filters
    across day boundaries before slicing today's data for output.

Public entry point: create_combined_l1_files()
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

from l1_filters import despike
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
        'T': 50000.0,
    }
    return thresholds.get(col, np.inf)


def _preferred_source(col, available_codes):
    """Determine ideal source given available satellites.

    Quality filtering is applied upstream (bad satellites removed from
    available_codes before this function is called), so no quality flag
    parameter is needed here.

    Parameters
    ----------
    col : str
        Variable name (e.g. 'Ux', 'rho').
    available_codes : list[int]
        Satellite codes with non-NaN, quality-passed values this minute.
    """
    if len(available_codes) == 1:
        return available_codes[0]

    # WIND is always top choice when present.
    if 3 in available_codes:
        return 3

    # ACE + DSCOVR only.
    if 1 in available_codes and 2 in available_codes:
        if col == 'rho':
            return 2          # DSCOVR density preferred.
        if col in ['Ux', 'Uy', 'Uz', 'T']:
            return 1

    return None


def _select_column_with_continuity(col, sat_series, bad_masks=None,
                                    use_hierarchy=False):
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
    use_hierarchy : bool
        When True (default) apply the priority rules (WIND > ACE > DSCOVR).
        When False, average all available quality-passed satellites so that
        no single spacecraft is preferred over another.
    """
    index = sat_series['ace'].index
    out_vals = np.full(len(index), np.nan, dtype=float)
    out_src = np.zeros(len(index), dtype=int)
    out_nsat = np.zeros(len(index), dtype=int)

    prev_value = np.nan

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

        if n_sat == 3:
            out_vals[i] = np.median([values[1], values[2], values[3]])
            out_src[i] = 0
            prev_value = out_vals[i]
            continue

        if n_sat == 2 and not use_hierarchy:
            out_vals[i] = np.mean(list(values.values()))
            out_src[i] = 0
            prev_value = out_vals[i]
            continue

        if n_sat == 2 and col in ['Bx', 'By', 'Bz'] and 1 in available and 2 in available:
            out_vals[i] = np.mean([values[1], values[2]])
            out_src[i] = 0
            prev_value = out_vals[i]
            continue

        candidate = _preferred_source(col, available)
        if candidate is None:
            candidate = available[0]

        # Continuity guard: if the candidate would jump too far from the
        # previous output value, pick whichever available satellite is
        # closest.
        if np.isfinite(prev_value) and abs(values[candidate] - prev_value) > _switch_threshold(col):
            closest = min(available, key=lambda c: abs(values[c] - prev_value))
            candidate = closest

        out_vals[i] = values[candidate]
        out_src[i] = candidate
        prev_value = out_vals[i]

    return pd.Series(out_vals, index=index), pd.Series(out_src, index=index), pd.Series(out_nsat, index=index)


def combine_data_priority(data_map, master_grid, use_hierarchy=False):
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

    cols = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']
    plasma_cols = {'Ux', 'Uy', 'Uz', 'rho', 'T'}
    df_combined = pd.DataFrame(index=master_grid, columns=cols)
    src_map = {}
    nsat_map = {}

    # Process each physical variable with priority + continuity rules.
    for col in cols:
        sat_series = {
            'ace': df_ace[col] if col in df_ace else pd.Series(np.nan, index=master_grid),
            'dscovr': df_dsc[col] if col in df_dsc else pd.Series(np.nan, index=master_grid),
            'wind': df_win[col] if col in df_win else pd.Series(np.nan, index=master_grid),
        }

        # Quality masks for plasma variables; None for magnetic field.
        col_masks = all_bad_masks if col in plasma_cols else None

        values, src_codes, n_sat = _select_column_with_continuity(
            col,
            sat_series,
            bad_masks=col_masks,
            use_hierarchy=use_hierarchy,
        )
        df_combined[col] = values
        src_map[col] = src_codes
        nsat_map[col] = n_sat

    # Interpolate short NaN gaps left by quality-gating.
    df_combined = df_combined.interpolate(
        method='time', limit=30, limit_area='inside')

    provenance = pd.DataFrame(index=master_grid)
    provenance['nSat'] = nsat_map['Ux'].astype('Int64')
    provenance['satUsed'] = src_map['Ux'].astype('Int64')

    return df_combined, provenance


def create_combined_l1_files(day, prev_day=None, next_day=None,
                             boundaries_re=(14, 32), use_hierarchy=False):
    """Build combined L1 products for *day* using rolling neighbour context.

    When *prev_day* and/or *next_day* are supplied the quality-scoring,
    satellite-selection, and despiking algorithms run over the full
    multi-day window so that day-boundary artefacts (cold-start of rolling
    filters, ``limit_growth`` warm-up, etc.) are eliminated.  Only the
    portion corresponding to *day* is written to disk.
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

    # ---- Build a master grid spanning the full context window ----
    window_start = pd.Timestamp(context_days[0])
    window_end = pd.Timestamp(context_days[-1]) + pd.Timedelta(days=1)
    n_minutes = int((window_end - window_start).total_seconds() / 60)
    master_grid = pd.date_range(start=window_start, periods=n_minutes,
                                freq='1min')

    # Quality-score + satellite-select over the full window.
    df_combined, provenance = combine_data_priority(data_map, master_grid,
                                                    use_hierarchy=use_hierarchy)

    # Despike over the full window so filters are warmed-up at day edges.
    if 'rho' in df_combined.columns or 'Ux' in df_combined.columns:
        df_combined = despike(df_combined)

    # ---- Slice out only *today* for file output ----
    today_start = pd.Timestamp(day)
    today_end = today_start + pd.Timedelta(days=1)
    today_mask = (df_combined.index >= today_start) & \
                 (df_combined.index < today_end)
    df_today = df_combined.loc[today_mask].copy()
    prov_today = provenance.loc[today_mask].copy()

    # Final pass: linearly interpolate any NaN gaps that exceed the
    # per-variable rolling fill performed inside combine_data_priority.
    df_today = df_today.interpolate(method='linear')

    # Write unpropagated combined file.
    outfile_comb = os.path.join(output_dir, 'L1_combined.dat')
    with open(outfile_comb, 'w', encoding='utf-8') as f:
        f.write(
            f'Combined L1 Data for {day} (Despiked, Unpropagated) (GSM nT, km/s, cm^-3, K)\n')
        f.write('year  mo  dy  hr  mn  sc msc Bx By Bz Ux Uy Uz rho T nSat satUsed\n')
        f.write('#START\n')
        for t, row in df_today.iterrows():
            if pd.isna(row['Bx']):
                continue
            n_sat_val = int(prov_today.at[t, 'nSat']) if pd.notna(
                prov_today.at[t, 'nSat']) else 0
            sat_used_val = int(prov_today.at[t, 'satUsed']) if pd.notna(
                prov_today.at[t, 'satUsed']) else 0
            f.write(
                f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} {t.second:2d} {t.microsecond//1000:3d} "
                f"{row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} "
                f"{row['Ux']:9.2f} {row['Uy']:9.2f} {row['Uz']:9.2f} "
                f"{row['rho']:9.4f} {row['T']:10.1f} {n_sat_val:2d} {sat_used_val:2d}\n"
            )
    print(f'Created {outfile_comb}')

    # Read mean ACE X position for travel-time propagation.
    pos_file = os.path.join(output_dir, 'L1_satpos.dat')
    mean_x_gse_km = 1.5e6

    # Fallback keeps pipeline running if position file is missing.
    if os.path.exists(pos_file):
        try:
            with open(pos_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                data_started = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('#START'):
                        data_started = True
                        continue
                    if not data_started:
                        continue
                    parts = line.split()
                    if len(parts) > 6:
                        ax_re = float(parts[6])
                        if not np.isnan(ax_re):
                            mean_x_gse_km = ax_re * 6371.0
                        break
        except Exception as e:
            print(
                f'    Warning: Could not read position file ({e}). Using default 1.5e6 km.')

    mock_orbit = pd.Series({'X_GSE': mean_x_gse_km})

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
