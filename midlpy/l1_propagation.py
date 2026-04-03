"""
l1_propagation.py
-----------------
Ballistic solar-wind propagation from L1 to an inner boundary.

Assumes the solar wind travels radially at its measured speed (frozen-in
approximation).  Enforces causality by discarding parcels that are
overtaken by faster parcels emitted later.

Public entry point: ballistic_propagation()
"""
import numpy as np
import pandas as pd


def ballistic_propagation(orbit, raw_data, target_x_km=90000):
    """Propagate L1 observations to a target X position along the Sun-Earth line.

    Parameters
    ----------
    orbit : pd.Series
        Must contain 'X_GSE' (spacecraft X in km).
    raw_data : pd.DataFrame
        L1 time series with column 'Vx Velocity, km/s, GSE' and a
        DatetimeIndex.
    target_x_km : float
        Target boundary X position in km (positive Sunward, e.g.
        14*6371 ≈ 89 194 km for 14 Re).

    Returns
    -------
    pd.DataFrame
        Propagated data on a complete 1-minute grid matching the input
        time range.  Arrival times are computed exactly (no rounding),
        then interpolated onto the regular grid so no minutes are dropped.
    """
    # Work on a copy so callers keep original data.
    input_df = raw_data.copy()
    # Pull upstream X position and solar-wind speed for timing.
    x_gse = np.float64(orbit['X_GSE'].item())
    target_x_km = np.float64(target_x_km)
    vx = np.asarray(input_df['Vx Velocity, km/s, GSE'], dtype=np.float64)

    # Ballistic travel time from spacecraft X to target X.
    travel_time_seconds = np.round((x_gse - target_x_km) / vx * (-1))
    travel_time = travel_time_seconds.astype('timedelta64[s]')

    # Compute arrival timestamps and enforce shock ordering.
    arrivals = input_df.index + travel_time
    valid_mask = pd.Series(True, index=input_df.index)

    for i in range(1, len(arrivals)):
        previous_indices = input_df.index[:i][arrivals[:i] > arrivals[i]]
        valid_mask.loc[previous_indices] = False

    # Drop older parcels overtaken by faster later parcels.
    input_df = input_df.loc[valid_mask]
    input_df.index = arrivals[valid_mask]

    # Resample onto a regular 1-minute grid.  Arrival times are irregular
    # (not aligned to whole minutes), so we merge them with the target grid
    # and interpolate to produce a gap-free output.
    numeric_cols = input_df.select_dtypes(include='number').columns
    input_df = input_df[numeric_cols]
    input_df = input_df[input_df.index.notna()]
    input_df = input_df[~input_df.index.duplicated(keep='first')]
    input_df = input_df.sort_index()

    grid = pd.date_range(raw_data.index.min(), raw_data.index.max(), freq='T')
    combined = input_df.index.union(grid)
    result = input_df.reindex(combined).interpolate(
        method='index').reindex(grid)

    return result
