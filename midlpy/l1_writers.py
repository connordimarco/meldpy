"""
l1_writers.py
-------------
Output formatters for MIDLResult objects.

Writes monthly CSV and DAT files to:
    output_dir/YYYY/MM/csv/{unpropagated,14Re,32Re}.csv
    output_dir/YYYY/MM/dat/{unpropagated,14Re,32Re}.dat

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
        output_dir/YYYY/MM/{csv,dat}/{unpropagated,14Re,32Re}.*

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

            _write_csv(group, label, year_month_dir)
            _write_dat(group, label, year_month_dir)
            n_months += 1

        print(f'Wrote {n_months} monthly files for {label} to {output_dir}/')


def _prepare_datasets(result):
    """Build dict of {label: DataFrame} from MIDLResult.

    For 'unpropagated', adds X_Re column using result.ref_x_re.
    """
    datasets = {}

    df_unp = result.unpropagated.copy()
    if result.ref_x_re:
        df_unp['X_Re'] = df_unp.index.map(
            lambda t: result.ref_x_re.get(t.date(), float('nan')))
    datasets['unpropagated'] = df_unp

    for b_re, df in result.propagated.items():
        datasets[f'{b_re}Re'] = df

    return datasets


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(df, label, year_month_dir):
    """Write one monthly CSV file."""
    out_dir = os.path.join(year_month_dir, 'csv')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{label}.csv')
    df.to_csv(path)


# ---------------------------------------------------------------------------
# DAT (space-delimited ASCII, SWMF/BATS-R-US compatible)
# ---------------------------------------------------------------------------

def _write_dat(df, label, year_month_dir):
    """Write one monthly DAT file."""
    out_dir = os.path.join(year_month_dir, 'dat')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{label}.dat')

    has_x_re = 'X_Re' in df.columns

    with open(path, 'w', encoding='utf-8') as f:
        # Header
        period = df.index[0].strftime('%Y-%m')
        f.write(f'MIDL {label} Data for {period} (GSM nT, km/s, cm^-3, K)\n')

        cols = 'year mo dy hr mn Bx By Bz Ux Uy Uz rho T'
        if has_x_re:
            cols += ' X_Re'
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

        for i in range(len(sub)):
            line = (
                f"{years[i]:4d} {months[i]:2d} {days[i]:2d} "
                f"{hours[i]:2d} {minutes[i]:2d} "
                f"{bx[i]:8.2f} {by[i]:8.2f} {bz[i]:8.2f} "
                f"{ux[i]:9.2f} {uy[i]:9.2f} {uz[i]:9.2f} "
                f"{rho[i]:9.4f} {temp[i]:10.1f}"
            )
            if has_x_re:
                line += f" {x_re[i]:9.2f}"
            lines.append(line)

        f.write('\n'.join(lines))
        f.write('\n')

