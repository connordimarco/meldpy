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
        Copy of *raw_data* re-indexed to arrival times at *target_x_km*,
        resampled to 1-minute cadence.
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
    # Snap arrivals to minute cadence used by downstream files.
    input_df['propagated_time'] = arrivals[valid_mask].round('min')
    # If multiple parcels land in the same minute, keep the fastest one.
    input_df = input_df.sort_values(
        by=['propagated_time', 'Vx Velocity, km/s, GSE'], ascending=[True, True])
    input_df = input_df.drop_duplicates(
        subset=['propagated_time'], keep='first')
    input_df.index = input_df['propagated_time']
    input_df = input_df.drop(columns=['propagated_time'])

    return input_df
