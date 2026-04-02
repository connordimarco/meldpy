"""
l1_midl.py
----------
Main entry point for the MIDL pipeline.

Processes an arbitrary date range of L1 solar wind data from L1_raw/
into merged, quality-screened, propagated output.

Public API: midl(start, end) -> MIDLResult
"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .l1_combine import combine_data_priority
from .l1_combine_T import combine_temperature
from .l1_filters import despike, interpolate_with_limits, smooth_transitions, INTERP_LIMITS
from .l1_propagation import ballistic_propagation
from .l1_readers import read_l1_data


_NUMERIC_COLS = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']


@dataclass
class MIDLResult:
    """Return value from midl().

    Attributes
    ----------
    unpropagated : pd.DataFrame
        Combined data at reference satellite position.
        Columns: Bx, By, Bz, Ux, Uy, Uz, rho, T.
        Index: DatetimeIndex at 1-minute cadence.
    propagated : dict[int, pd.DataFrame]
        Ballistically propagated data keyed by boundary distance in Re.
        Default keys: 14, 32. Same columns as unpropagated.
    ref_x_re : dict[datetime.date, float]
        X_GSM position (in Earth radii) of the reference satellite for each
        calendar day.  The reference satellite is the one closest to Earth.
    """
    unpropagated: pd.DataFrame
    propagated: dict
    ref_x_re: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day_range(start, end):
    """Yield 'YYYY-MM-DD' strings for each day in [start, end] inclusive."""
    current = start
    while current <= end:
        yield current.strftime('%Y-%m-%d')
        current += timedelta(days=1)


def _load_raw_range(raw_dir, start, end):
    """Load L1_raw .dat files for [start, end] into per-satellite DataFrames.

    Returns dict: {'ace': df, 'dscovr': df, 'wind': df}.
    Satellites with no data are omitted.
    """
    data_map = {}
    for sat in ('ace', 'dscovr', 'wind'):
        frames = []
        for day_str in _day_range(start, end):
            dt = datetime.strptime(day_str, '%Y-%m-%d')
            path = os.path.join(raw_dir, dt.strftime('%Y/%m/%d'),
                                f'L1_{sat}.dat')
            df = read_l1_data(path)
            if not df.empty:
                frames.append(df[_NUMERIC_COLS])
        if frames:
            combined = pd.concat(frames).sort_index()
            data_map[sat] = combined[~combined.index.duplicated(keep='first')]
    return data_map


def _read_sat_positions(pos_file):
    """Read per-satellite noon X positions in km from L1_satpos.dat."""
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


def _load_positions_range(raw_dir, start, end):
    """Load satellite positions for [start, end] into a dict indexed by date.

    Reads L1_satpos.dat from the same raw_dir tree as the satellite data.
    Returns dict: {date -> {'ace': x_km, 'dscovr': x_km, 'wind': x_km}}.
    """
    positions = {}
    for day_str in _day_range(start, end):
        dt = datetime.strptime(day_str, '%Y-%m-%d')
        pos_file = os.path.join(raw_dir, dt.strftime('%Y/%m/%d'),
                                'L1_satpos.dat')
        positions[dt.date()] = _read_sat_positions(pos_file)
    return positions


def _propagate_to_reference(data_map, positions):
    """Shift satellites to daily reference position (closest to Earth).

    Modifies data_map in place. Returns a dict {date -> x_ref_km} for use
    in final propagation to boundary.
    """
    ref_x_daily = {}

    # Collect all dates that have data.
    all_dates = set()
    for sat_df in data_map.values():
        all_dates.update(sat_df.index.date)
    all_dates = sorted(all_dates)

    # Forward-fill positions for days with missing satpos files.
    last_good_pos = None

    for date in all_dates:
        pos = positions.get(date)
        if pos is None:
            pos = last_good_pos if last_good_pos else {
                'ace': np.nan, 'dscovr': np.nan, 'wind': np.nan}
        if any(np.isfinite(v) for v in pos.values()):
            last_good_pos = pos

        available_x = {sat: pos[sat] for sat in data_map
                       if np.isfinite(pos.get(sat, np.nan))}

        if not available_x:
            ref_x_daily[date] = 1.5e6
            continue

        ref_sat = min(available_x, key=lambda s: available_x[s])
        x_ref_km = available_x[ref_sat]
        ref_x_daily[date] = x_ref_km

        # Shift non-reference satellites for this day's data.
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)

        for sat in list(data_map.keys()):
            x_sat = available_x.get(sat, np.nan)
            if not np.isfinite(x_sat) or x_sat <= x_ref_km:
                continue

            day_mask = ((data_map[sat].index >= day_start) &
                        (data_map[sat].index < day_end))
            if not day_mask.any():
                continue

            df_day = data_map[sat].loc[day_mask].copy()
            df_day = df_day.rename(
                columns={'Ux': 'Vx Velocity, km/s, GSE'})
            orbit = pd.Series({'X_GSE': x_sat})
            df_prop = ballistic_propagation(
                orbit, df_day, target_x_km=x_ref_km)
            df_prop = df_prop.rename(
                columns={'Vx Velocity, km/s, GSE': 'Ux'})

            # Replace this day's slice in the full DataFrame.
            data_map[sat] = pd.concat([
                data_map[sat].loc[~day_mask],
                df_prop
            ]).sort_index()

    return ref_x_daily


def _compute_source_changed(source_map):
    """Build per-column boolean mask: True where satellite source changed."""
    source_changed = {}
    for col, src in source_map.items():
        vals = src.values
        changed = np.zeros(len(vals), dtype=bool)
        for k in range(1, len(vals)):
            if vals[k] is not None and vals[k - 1] is not None:
                changed[k] = vals[k] != vals[k - 1]
        source_changed[col] = pd.Series(changed, index=src.index)
    return source_changed


def _propagate_to_boundary(df_combined, ref_x_daily, target_km):
    """Propagate combined data to a fixed boundary using per-day reference X."""
    all_dates = sorted(set(df_combined.index.date))
    frames = []

    for date in all_dates:
        x_ref = ref_x_daily.get(date, 1.5e6)
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)
        day_mask = ((df_combined.index >= day_start) &
                    (df_combined.index < day_end))
        df_day = df_combined.loc[day_mask].copy()

        if df_day.empty or df_day['Ux'].isna().all():
            frames.append(df_day)
            continue

        df_day = df_day.rename(
            columns={'Ux': 'Vx Velocity, km/s, GSE'})
        orbit = pd.Series({'X_GSE': x_ref})
        df_prop = ballistic_propagation(
            orbit, df_day, target_x_km=target_km)
        df_prop = df_prop.rename(
            columns={'Vx Velocity, km/s, GSE': 'Ux'})
        frames.append(df_prop)

    if not frames:
        return df_combined.copy()
    return pd.concat(frames).sort_index()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def midl(start, end, raw_dir='L1_raw', boundaries_re=(14, 32)):
    """Process L1 solar wind data for [start, end].

    Reads raw satellite data and position files from raw_dir/, applies
    the full pipeline (despike, interpolate, propagate-to-reference,
    quality-score, source-select, combine, smooth, propagate-to-boundary),
    and returns
    a MIDLResult.

    Parameters
    ----------
    start, end : str or pd.Timestamp
        Date range to process (inclusive), e.g. '2024-05-09', '2024-05-11'.
    raw_dir : str
        Path to directory tree containing per-day satellite data and
        L1_satpos.dat files (raw_dir/YYYY/MM/DD/).
    boundaries_re : tuple of int
        Propagation target distances in Earth radii. Default (14, 32).

    Returns
    -------
    MIDLResult
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Pad by 1 day for boundary context (replaces old 3-day window).
    load_start = (start - pd.Timedelta(days=1)).normalize()
    load_end = (end + pd.Timedelta(days=1)).normalize()

    # Stage 0: Load raw data.
    print(f'Loading L1_raw data for {load_start.date()} to {load_end.date()}...')
    data_map = _load_raw_range(raw_dir, load_start, load_end)

    if not data_map:
        print('No satellite data found.')
        empty = pd.DataFrame(columns=_NUMERIC_COLS)
        return MIDLResult(
            unpropagated=empty,
            propagated={b: empty.copy() for b in boundaries_re},
            ref_x_re={},
        )

    # Stage 1: Despike.
    print('Despiking...')
    for sat in data_map:
        data_map[sat] = despike(data_map[sat])

    # Stage 2: Interpolate per-satellite gaps.
    print('Interpolating gaps...')
    for sat in data_map:
        data_map[sat] = interpolate_with_limits(data_map[sat], INTERP_LIMITS)

    # Stage 3: Propagate to reference position.
    print('Propagating to reference positions...')
    positions = _load_positions_range(raw_dir, load_start, load_end)
    ref_x_daily = _propagate_to_reference(data_map, positions)

    # Deduplicate after propagation — time-shifting can create collisions at
    # day boundaries. Keep the fastest parcel (most negative Ux) per minute.
    for sat in data_map:
        df = data_map[sat]
        if df.index.duplicated().any():
            df = df.sort_values('Ux', ascending=True)
            data_map[sat] = df[~df.index.duplicated(keep='first')]

    # Build master grid spanning the full padded window.
    grid_start = load_start
    grid_end = load_end + pd.Timedelta(days=1)
    n_minutes = int((grid_end - grid_start).total_seconds() / 60)
    master_grid = pd.date_range(start=grid_start, periods=n_minutes,
                                freq='1min')

    # Stage 4: Quality score + source select.
    print('Running quality scoring and source selection...')
    df_combined, _provenance, source_map = combine_data_priority(
        data_map, master_grid)

    print('Combining temperature...')
    df_combined['T'] = combine_temperature(data_map, master_grid)

    # Stage 5: Smooth transitions.
    print('Smoothing transitions...')
    source_changed = _compute_source_changed(source_map)
    df_combined = smooth_transitions(
        df_combined, source_changed=source_changed)

    # Final interpolation pass.
    df_combined = df_combined.interpolate(method='linear')

    # Stage 6: Propagate to boundaries.
    propagated = {}
    for b_re in boundaries_re:
        target_km = b_re * 6371.0
        print(f'Propagating to {b_re} Re ({target_km:.0f} km)...')
        propagated[b_re] = _propagate_to_boundary(
            df_combined, ref_x_daily, target_km)

    # Stage 7: Slice to requested range and return.
    result_start = start.normalize()
    result_end = (end + pd.Timedelta(days=1)).normalize()
    mask = ((df_combined.index >= result_start) &
            (df_combined.index < result_end))

    result_propagated = {}
    for b_re, df_prop in propagated.items():
        prop_mask = ((df_prop.index >= result_start) &
                     (df_prop.index < result_end))
        result_propagated[b_re] = df_prop.loc[prop_mask].copy()

    # Convert reference positions from km back to Re.
    ref_x_re = {date: x_km / 6371.0 for date, x_km in ref_x_daily.items()}

    print('Done.')
    return MIDLResult(
        unpropagated=df_combined.loc[mask].copy(),
        propagated=result_propagated,
        ref_x_re=ref_x_re,
    )
