"""
l1_pipeline.py
--------------
Per-satellite download, processing, and L1 file creation.

Each call to get_one_day_swmf_input() processes one UTC day for all three
satellites:
  - Downloads raw CDF/NC data from CDAWeb (ACE, WIND) and NOAA NGDC (DSCOVR).
  - Resamples to a 1-minute master grid.
  - Rotates coordinate frames to GSM where needed.
  - Converts WIND thermal speed to temperature in Kelvin.
  - Writes per-satellite L1_*.dat files.

Public entry points:
  get_one_day_swmf_input(day, cda)  — download + process all satellites.
  create_position_file(day, cda)    — write L1_satpos.dat (noon positions).
"""
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

from l1_coordinates import gse_to_gsm
from l1_downloaders import (
    download_cdaweb_files,
    download_dscovr_ngdc,
    download_position_cdaweb_files,
)
from l1_readers import cdf_to_df, nc_gz_to_df


_NGDC_F1M_VARS = {
    'proton_vx_gsm': ['Ux'],
    'proton_vy_gsm': ['Uy'],
    'proton_vz_gsm': ['Uz'],
    'proton_density': ['rho'],
    'proton_temperature': ['T'],
}

_NGDC_M1M_VARS = {
    'bx_gsm': ['Bx'],
    'by_gsm': ['By'],
    'bz_gsm': ['Bz'],
}


def process_satellite(
    sat_name,
    mag_file_pattern,
    plasma_file_pattern,
    var_map,
    data_dir,
    trange_start,
    trange_end,
    cleanup_cdfs=True,
):
    """Process one CDAWeb-sourced satellite for a single day.

    Reads separate MAG and plasma CDF files, resamples both to 1-minute,
    applies any required coordinate rotation and temperature conversion,
    then writes a per-satellite L1_<sat_name>.dat file.

    Parameters
    ----------
    sat_name : str  ('ace' | 'wind')
    mag_file_pattern : str  Glob prefix for the magnetometer CDF (e.g. 'ac_h0_mfi').
    plasma_file_pattern : str  Glob prefix for the plasma CDF (e.g. 'ac_h0_swe').
    var_map : dict  Variable-name mapping (see get_one_day_swmf_input for examples).
    data_dir : str  Root scratch directory where CDFs were downloaded.
    trange_start, trange_end : str  'YYYY-MM-DD' day boundaries for the output grid.
    cleanup_cdfs : bool  Remove raw CDF files after writing output (default True).
    """
    print(f'\nProcessing {sat_name}...')

    # Find downloaded MAG/PLASMA CDF files for this satellite/day.
    mag_files = glob.glob(
        f'{data_dir}/**/{mag_file_pattern}*.cdf', recursive=True)
    plasma_files = glob.glob(
        f'{data_dir}/**/{plasma_file_pattern}*.cdf', recursive=True)

    if not mag_files and not plasma_files:
        print(f'Missing both MAG and Plasma files for {sat_name}. Skipping.')
        return

    # Build empty placeholders when one stream is missing.
    if mag_files:
        df_mag = cdf_to_df(
            mag_files[0], var_map['mag_time'], var_map['mag_vars'])
    else:
        print(f'Missing MAG files for {sat_name}. Filling with NaNs.')
        mag_cols = []
        for cols in var_map['mag_vars'].values():
            mag_cols.extend(cols)
        df_mag = pd.DataFrame(columns=mag_cols)

    if plasma_files:
        df_plasma = cdf_to_df(
            plasma_files[0], var_map['plasma_time'], var_map['plasma_vars'])
    else:
        print(f'Missing Plasma files for {sat_name}. Filling with NaNs.')
        plasma_cols = []
        for cols in var_map['plasma_vars'].values():
            plasma_cols.extend(cols)
        df_plasma = pd.DataFrame(columns=plasma_cols)

    # Resample both streams onto one 1-minute master grid.
    grid = pd.date_range(start=trange_start, end=trange_end, freq='1min')
    df_master = pd.DataFrame(index=grid)

    if not df_mag.empty:
        df_mag_res = df_mag.resample(
            '1min').mean().interpolate(method='time', limit=1)
    else:
        df_mag_res = pd.DataFrame(columns=df_mag.columns)

    if not df_plasma.empty:
        df_plasma_res = df_plasma.resample(
            '1min').mean().interpolate(method='time', limit=1)
    else:
        df_plasma_res = pd.DataFrame(columns=df_plasma.columns)

    df_final = df_master.join(df_mag_res).join(df_plasma_res)

    # Rotate vectors only where source coordinates require it.
    if sat_name == 'dscovr':
        print('Rotating DSCOVR data from GSE to GSM...')
        gse_to_gsm(df_final, ['Bx', 'By', 'Bz'])
        gse_to_gsm(df_final, ['Ux', 'Uy', 'Uz'])
    elif sat_name == 'wind':
        print('Rotating WIND Plasma data from GSE to GSM...')
        gse_to_gsm(df_final, ['Ux', 'Uy', 'Uz'])

    # WIND temperature comes in as thermal speed; convert to Kelvin.
    if 'v_th' in df_final.columns:
        df_final['T'] = df_final['v_th'] ** 2 * 60.5
    elif 'T' not in df_final.columns:
        df_final['T'] = np.nan

    # Keep obviously bad temperature fill values out of output.
    if 'T' in df_final.columns:
        df_final.loc[df_final['T'] > 1e9, 'T'] = np.nan

    dt_start = datetime.strptime(trange_start, '%Y-%m-%d')
    output_dir = dt_start.strftime('L1/%Y/%m/%d')
    os.makedirs(output_dir, exist_ok=True)

    # Write per-satellite L1 text file.
    output_file = os.path.join(output_dir, f'L1_{sat_name}.dat')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'Produced from {sat_name} CDAWeb CDFs (nT, km/s, cm^-3, K)\n')
        f.write('year  mo  dy  hr  mn  sc msc Bx By Bz Ux Uy Uz rho T\n')
        f.write('#START\n')

        for t, row in df_final.iterrows():
            if pd.isna(row['Bx']):
                continue

            f.write(
                f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} {t.second:2d} {t.microsecond//1000:3d} "
                f"{row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} "
                f"{row['Ux']:9.2f} {row['Uy']:9.2f} {row['Uz']:9.2f} "
                f"{row['rho']:9.4f} {row['T']:10.1f}\n"
            )

    print(f'Saved {output_file}')

    # Remove raw CDFs once we have daily output files.
    if cleanup_cdfs:
        print(f'Cleaning up {sat_name} CDFs...')
        for fpath in mag_files + plasma_files:
            try:
                os.remove(fpath)
            except Exception as e:
                print(f'Could not remove {fpath}: {e}')


def process_satellite_ngdc(day, data_dir, trange_start, trange_end, cleanup=True):
    """Process DSCOVR using NOAA NGDC 1-minute products.

    Downloads f1m (plasma) and m1m (magnetometer) gzipped NetCDF files,
    resamples to 1-minute, and writes L1_dscovr.dat.
    NGDC data is already in GSM so no rotation is needed.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    data_dir : str  Scratch directory for temporary downloads.
    trange_start, trange_end : str  Day boundaries for the output grid.
    cleanup : bool  Remove downloaded .nc.gz files after writing (default True).
    """
    print('\nProcessing DSCOVR (NGDC)...')

    # Download 1-minute plasma + mag products from NGDC.
    paths = download_dscovr_ngdc(day, data_dir)

    if 'f1m' in paths:
        df_plasma = nc_gz_to_df(paths['f1m'], 'time', _NGDC_F1M_VARS)
    else:
        print('  Missing f1m (plasma) — DSCOVR plasma will be NaN.')
        df_plasma = pd.DataFrame(columns=['Ux', 'Uy', 'Uz', 'rho', 'T'])

    if 'm1m' in paths:
        df_mag = nc_gz_to_df(paths['m1m'], 'time', _NGDC_M1M_VARS)
    else:
        print('  Missing m1m (mag) — DSCOVR mag will be NaN.')
        df_mag = pd.DataFrame(columns=['Bx', 'By', 'Bz'])

    # Resample to the same 1-minute master grid as other satellites.
    grid = pd.date_range(start=trange_start, end=trange_end, freq='1min')
    df_master = pd.DataFrame(index=grid)

    if not df_mag.empty:
        df_mag_res = df_mag.resample(
            '1min').mean().interpolate(method='time', limit=1)
    else:
        df_mag_res = pd.DataFrame(columns=['Bx', 'By', 'Bz'])

    if not df_plasma.empty:
        df_plasma_res = df_plasma.resample(
            '1min').mean().interpolate(method='time', limit=1)
    else:
        df_plasma_res = pd.DataFrame(columns=['Ux', 'Uy', 'Uz', 'rho', 'T'])

    df_final = df_master.join(df_mag_res).join(df_plasma_res)

    # NGDC temperature is already Kelvin; just guard bad outliers.
    if 'T' not in df_final.columns:
        df_final['T'] = np.nan

    if 'T' in df_final.columns:
        df_final.loc[df_final['T'] > 1e9, 'T'] = np.nan

    dt_start = datetime.strptime(trange_start, '%Y-%m-%d')
    output_dir = dt_start.strftime('L1/%Y/%m/%d')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'L1_dscovr.dat')

    # Write DSCOVR daily L1 file.
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Produced from DSCOVR NOAA NGDC NetCDF (nT, km/s, cm^-3, K)\n')
        f.write('year  mo  dy  hr  mn  sc msc Bx By Bz Ux Uy Uz rho T\n')
        f.write('#START\n')
        for t, row in df_final.iterrows():
            if pd.isna(row.get('Bx', np.nan)):
                continue
            f.write(
                f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} "
                f"{t.second:2d} {t.microsecond//1000:3d} "
                f"{row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} "
                f"{row['Ux']:9.2f} {row['Uy']:9.2f} {row['Uz']:9.2f} "
                f"{row['rho']:9.4f} {row['T']:10.1f}\n"
            )

    print(f'Saved {output_file}')

    # Delete temporary NGDC downloads when requested.
    if cleanup:
        print('  Cleaning up DSCOVR NGDC files...')
        for fpath in paths.values():
            try:
                os.remove(fpath)
            except Exception as e:
                print(f'  Could not remove {fpath}: {e}')


def create_position_file(day, cda, cleanup_cdfs=True):
    """Write L1_satpos.dat containing mean noon GSM positions for all three satellites.

    Downloads a narrow 11:00–13:00 UT window of orbit data, averages the
    X/Y/Z positions, and writes a single-row file used by the propagator to
    determine ballistic travel time.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    cda : pyspedas.CDAWeb
    cleanup_cdfs : bool  Remove CDF files after writing (default True).
    """
    print(f'\n--- Generating L1_satpos.dat for {day} ---')

    # Position output is one noon-time row per day.
    dt_day = pd.to_datetime(day)
    data_dir = 'cdf_temp'

    dt_start = datetime.strptime(day, '%Y-%m-%d')
    output_dir = dt_start.strftime('L1/%Y/%m/%d')
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, 'L1_satpos.dat')

    # Pull a narrow 11:00-13:00 UT window for stable mean position.
    urllist = download_position_cdaweb_files(cda, day, data_dir)

    if not urllist:
        print('No position files found.')
        return

    # ACE position is GSM in km, so convert to Re.
    ace_files = glob.glob(f'{data_dir}/**/ac_h0_mfi_*.cdf', recursive=True)
    if ace_files:
        df_ace = cdf_to_df(ace_files[0], 'Epoch', {
                           'SC_pos_GSM': ['Ax', 'Ay', 'Az']})
        df_ace[['Ax', 'Ay', 'Az']] /= 6371.0
        ace_mean = df_ace[['Ax', 'Ay', 'Az']].mean()
    else:
        ace_mean = pd.Series({'Ax': np.nan, 'Ay': np.nan, 'Az': np.nan})

    # WIND position is already GSM in Re.
    wind_files = glob.glob(f'{data_dir}/**/wi_h0_mfi_*.cdf', recursive=True)
    if wind_files:
        df_wind = cdf_to_df(wind_files[0], 'Epoch', {
                            'PGSM': ['Wx', 'Wy', 'Wz']})
        wind_mean = df_wind[['Wx', 'Wy', 'Wz']].mean()
    else:
        wind_mean = pd.Series({'Wx': np.nan, 'Wy': np.nan, 'Wz': np.nan})

    # DSCOVR position comes in GSE km; convert to Re then rotate to GSM.
    dsc_files = glob.glob(
        f'{data_dir}/**/dscovr_orbit_pre_*.cdf', recursive=True)
    if dsc_files:
        df_dsc = cdf_to_df(dsc_files[0], 'Epoch', {
                           'GSE_POS': ['Dx', 'Dy', 'Dz']})
        df_dsc[['Dx', 'Dy', 'Dz']] /= 6371.0
        gse_to_gsm(df_dsc, ['Dx', 'Dy', 'Dz'])
        dsc_mean = df_dsc[['Dx', 'Dy', 'Dz']].mean()
    else:
        dsc_mean = pd.Series({'Dx': np.nan, 'Dy': np.nan, 'Dz': np.nan})

    # Write one merged row for downstream propagation logic.
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(
            f'Multi-Satellite Position File (GSM Coordinates, Re) for {day}\n')
        f.write('year  mo  dy  hr  mn  sc  Ax  Ay  Az  Dx  Dy  Dz  Wx  Wy  Wz\n')
        f.write('#START\n')

        def fmt(val):
            return f'{val:8.1f}' if pd.notna(val) else '     nan'

        line = (
            f"{dt_day.year:4d} {dt_day.month:2d} {dt_day.day:2d} 12  0  0 "
            f"{fmt(ace_mean['Ax'])} {fmt(ace_mean['Ay'])} {fmt(ace_mean['Az'])} "
            f"{fmt(dsc_mean['Dx'])} {fmt(dsc_mean['Dy'])} {fmt(dsc_mean['Dz'])} "
            f"{fmt(wind_mean['Wx'])} {fmt(wind_mean['Wy'])} {fmt(wind_mean['Wz'])}\n"
        )
        f.write(line)

    print(f'Saved {output_filepath}')

    # Cleanup raw position files after writing output.
    if cleanup_cdfs:
        for fpath in ace_files + wind_files + dsc_files:
            try:
                os.remove(fpath)
            except Exception:
                pass


def get_one_day_swmf_input(day, cda):
    """Download and process all three satellites for a single UTC day.

    Downloads ACE and WIND CDF data via CDAWeb and DSCOVR 1-minute products
    from NOAA NGDC, then calls per-satellite processors to write
    L1_ace.dat, L1_dscovr.dat, and L1_wind.dat into L1/YYYY/MM/DD/.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    cda : pyspedas.CDAWeb
    """
    # Process one UTC day end-to-end for ACE/DSCOVR/WIND.
    trange_start = day
    trange_end = (pd.to_datetime(day) +
                  pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    data_dir = 'cdf_temp'

    # Download ACE+WIND CDF sources via CDAWeb.
    datasets = ['AC_H0_MFI', 'AC_H0_SWE', 'WI_H0_MFI', 'WI_H1_SWE']

    download_cdaweb_files(cda, datasets, trange_start, trange_end, data_dir)

    # Variable maps define how source variables land in L1 columns.
    ace_map = {
        'mag_time': 'Epoch',
        'mag_vars': {'BGSM': ['Bx', 'By', 'Bz']},
        'plasma_time': 'Epoch',
        'plasma_vars': {'V_GSM': ['Ux', 'Uy', 'Uz'], 'Np': ['rho'], 'Tpr': ['T']},
    }

    win_map = {
        'mag_time': 'Epoch',
        'mag_vars': {'BGSM': ['Bx', 'By', 'Bz']},
        'plasma_time': 'Epoch',
        'plasma_vars': {
            'Proton_VX_moment': ['Ux'],
            'Proton_VY_moment': ['Uy'],
            'Proton_VZ_moment': ['Uz'],
            'Proton_Np_moment': ['rho'],
            'Proton_W_moment': ['v_th'],
        },
    }

    # Run per-satellite processing and write individual L1 files.
    process_satellite('ace', 'ac_h0_mfi', 'ac_h0_swe',
                      ace_map, data_dir, trange_start, trange_end)
    process_satellite_ngdc(day, data_dir, trange_start, trange_end)
    process_satellite('wind', 'wi_h0_mfi', 'wi_h1_swe',
                      win_map, data_dir, trange_start, trange_end)
