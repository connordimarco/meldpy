"""
l1_writers.py
-------------
Output formatters for MIDLResult objects.

Writes monthly CSV and DAT files to:
    output_dir/YYYY/MM/csv/YYYYMM_{unpropagated,14Re,32Re}.csv
    output_dir/YYYY/MM/dat/YYYYMM_{unpropagated,14Re,32Re}.dat

Usage:
    from midlpy import midl, write_monthly_outputs

    result = midl('2024-05-09', '2024-05-11')
    write_monthly_outputs(result, output_dir='data')
"""
import os

import pandas as pd


def write_monthly_outputs(result, output_dir='data'):
    """Write MIDLResult to monthly CSV and DAT files.

    Creates directory structure:
        output_dir/YYYY/MM/{csv,dat}/YYYYMM_{unpropagated,14Re,32Re}.*

    For unpropagated data, an X_Re column is added containing the X_GSM
    distance (in Earth radii) of the reference satellite for each day.

    Parameters
    ----------
    result : MIDLResult
    output_dir : str
    """
    datasets = _prepare_datasets(result)

    for label, df in datasets.items():
        if df.empty:
            continue

        df = df.copy()
        df['_ym'] = df.index.to_period('M')

        n_months = 0
        for period, group in df.groupby('_ym'):
            group = group.drop(columns='_ym')
            group.index.name = 'timestamp'

            year_month_dir = os.path.join(
                output_dir, f'{period.year:04d}', f'{period.month:02d}')

            ym_prefix = f'{period.year:04d}{period.month:02d}'
            _write_csv(group, label, year_month_dir, ym_prefix)
            _write_dat(group, label, year_month_dir, ym_prefix)
            n_months += 1

        print(f'Wrote {n_months} monthly files for {label} to {output_dir}/')


def _frozenset_to_str(fs):
    """Convert frozenset of satellite codes to a sorted digit string.

    1=ACE, 2=DSCOVR, 3=WIND.  e.g. frozenset({1,3}) -> '13'.
    """
    if fs is None or (isinstance(fs, frozenset) and not fs):
        return ''
    return ''.join(str(c) for c in sorted(fs))


# Map from source_map variable keys to output column names.
_SOURCE_COLUMNS = {
    'B_source':   'Bx',    # Bx/By/Bz all share the same source
    'Ux_source':  'Ux',
    'Uyz_source': 'Uy',    # Uy/Uz share the same source
    'rho_source': 'rho',
    'T_source':   'T',
}


def _prepare_datasets(result):
    """Build dict of {label: DataFrame} from MIDLResult.

    For 'unpropagated', adds X_Re and source columns using result metadata.
    """
    datasets = {}

    df_unp = result.unpropagated.copy()
    if result.ref_x_re:
        df_unp['X_Re'] = df_unp.index.map(
            lambda t: result.ref_x_re.get(t.date(), float('nan')))

    # Add source provenance columns.
    if result.source_map:
        for out_col, src_key in _SOURCE_COLUMNS.items():
            if src_key in result.source_map:
                src = result.source_map[src_key].reindex(df_unp.index)
                df_unp[out_col] = src.map(
                    lambda fs: _frozenset_to_str(fs) or float('nan'))

    datasets['unpropagated'] = df_unp

    for b_re, df in result.propagated.items():
        datasets[f'{b_re}Re'] = df

    return datasets


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_CSV_PRECISION = {
    'Bx': '%.2f', 'By': '%.2f', 'Bz': '%.2f',
    'Ux': '%.1f', 'Uy': '%.2f', 'Uz': '%.2f',
    'rho': '%.3f', 'T': '%.0f', 'X_Re': '%.2f',
}


def _write_csv(df, label, year_month_dir, ym_prefix):
    """Write one monthly CSV file with precision matching DAT output."""
    out_dir = os.path.join(year_month_dir, 'csv')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{ym_prefix}_{label}.csv')
    rounded = df.copy()
    for col, spec in _CSV_PRECISION.items():
        if col in rounded.columns:
            decimals = int(spec.split('.')[1].rstrip('f'))
            rounded[col] = rounded[col].round(decimals)
    rounded.to_csv(path)


# ---------------------------------------------------------------------------
# DAT (space-delimited ASCII, SWMF/BATS-R-US compatible)
# ---------------------------------------------------------------------------

def _write_dat(df, label, year_month_dir, ym_prefix):
    """Write one monthly DAT file."""
    out_dir = os.path.join(year_month_dir, 'dat')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{ym_prefix}_{label}.dat')

    has_x_re = 'X_Re' in df.columns
    source_cols = [c for c in ('B_source', 'Ux_source', 'Uyz_source',
                               'rho_source', 'T_source') if c in df.columns]

    with open(path, 'w', encoding='utf-8') as f:
        # Header
        period = df.index[0].strftime('%Y-%m')
        f.write(f'MIDL {label} Data for {period} (GSM nT, km/s, cm^-3, K)\n')

        cols = 'year month day hour minute Bx By Bz Ux Uy Uz rho T'
        if has_x_re:
            cols += ' X_Re'
        for sc in source_cols:
            cols += f' {sc}'
        f.write(cols + '\n')
        f.write('#START\n')

        # Vectorized writing: build string array from numpy
        valid = df['Bx'].notna()
        sub = df.loc[valid]

        if sub.empty:
            return

        ts = sub.index
        lines = []
        years = ts.year
        months = ts.month
        days = ts.day
        hours = ts.hour
        minutes = ts.minute

        bx = sub['Bx'].values
        by = sub['By'].values
        bz = sub['Bz'].values
        ux = sub['Ux'].values
        uy = sub['Uy'].values
        uz = sub['Uz'].values
        rho = sub['rho'].values
        temp = sub['T'].values
        x_re = sub['X_Re'].values if has_x_re else None
        source_vals = {sc: sub[sc].values for sc in source_cols}

        for i in range(len(sub)):
            line = (
                f"{years[i]:4d} {months[i]:2d} {days[i]:2d} "
                f"{hours[i]:2d} {minutes[i]:2d} "
                f"{bx[i]:8.2f} {by[i]:8.2f} {bz[i]:8.2f} "
                f"{ux[i]:9.1f} {uy[i]:9.2f} {uz[i]:9.2f} "
                f"{rho[i]:9.3f} {temp[i]:10.0f}"
            )
            if has_x_re:
                line += f" {x_re[i]:9.2f}"
            for sc in source_cols:
                v = source_vals[sc][i]
                line += f" {str(v) if isinstance(v, str) else 'NaN':>4s}"
            lines.append(line)

        f.write('\n'.join(lines))
        f.write('\n')

