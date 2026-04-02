"""
l1_writers.py
-------------
Output formatters for MIDLResult objects.

Usage:
    from midlpy import midl, write_monthly_parquet, write_daily_dat

    result = midl('2024-05-09', '2024-05-11')
    write_monthly_parquet(result, output_dir='L1_db/data')
    write_daily_dat(result, output_dir='L1')
"""
import os

import pandas as pd


def write_monthly_parquet(result, output_dir, targets=None):
    """Write MIDLResult to monthly Parquet files.

    Creates directory structure:
        output_dir/unpropagated/IMF_YYYY_MM.parquet
        output_dir/14re/IMF_YYYY_MM.parquet
        output_dir/32re/IMF_YYYY_MM.parquet

    Parameters
    ----------
    result : MIDLResult
    output_dir : str
    targets : list of str, optional
        Which datasets to write. Default: all (unpropagated + all propagated).
    """
    datasets = {}
    datasets['unpropagated'] = result.unpropagated

    for b_re, df in result.propagated.items():
        datasets[f'{b_re}re'] = df

    if targets:
        datasets = {k: v for k, v in datasets.items() if k in targets}

    for label, df in datasets.items():
        if df.empty:
            continue

        sub_dir = os.path.join(output_dir, label)
        os.makedirs(sub_dir, exist_ok=True)

        # Group by year-month and write one Parquet per month.
        df = df.copy()
        df['_ym'] = df.index.to_period('M')

        for period, group in df.groupby('_ym'):
            group = group.drop(columns='_ym')
            group.index.name = 'timestamp'
            fname = f'IMF_{period.year:04d}_{period.month:02d}.parquet'
            path = os.path.join(sub_dir, fname)
            group.to_parquet(path, engine='pyarrow')

        print(f'Wrote {len(df.groupby("_ym"))} monthly files to {sub_dir}/')

    # Write reference position alongside unpropagated data.
    if result.ref_x_re and (not targets or 'unpropagated' in targets):
        write_ref_position_parquet(result, output_dir)


def write_ref_position_parquet(result, output_dir):
    """Write reference satellite X_GSM position to a single Parquet file.

    Creates:
        output_dir/unpropagated/ref_position.parquet

    Columns: date (datetime64), X_GSM_Re (float64).

    Parameters
    ----------
    result : MIDLResult
    output_dir : str
    """
    if not result.ref_x_re:
        return

    df = pd.DataFrame([
        {'date': pd.Timestamp(date), 'X_GSM_Re': x_re}
        for date, x_re in sorted(result.ref_x_re.items())
    ])

    sub_dir = os.path.join(output_dir, 'unpropagated')
    os.makedirs(sub_dir, exist_ok=True)
    path = os.path.join(sub_dir, 'ref_position.parquet')
    df.to_parquet(path, engine='pyarrow', index=False)
    print(f'Wrote reference positions to {path}')


def write_daily_dat(result, output_dir='L1'):
    """Write MIDLResult to per-day .dat files (backward compatible).

    Creates:
        output_dir/YYYY/MM/DD/L1_combined.dat  (unpropagated)
        output_dir/YYYY/MM/DD/IMF_14Re.dat
        output_dir/YYYY/MM/DD/IMF_32Re.dat

    Parameters
    ----------
    result : MIDLResult
    output_dir : str
    """
    # Write unpropagated.
    _write_daily_frames(
        result.unpropagated, output_dir,
        filename='L1_combined.dat',
        header_template='Combined L1 Data for {day} (Unpropagated) (GSM nT, km/s, cm^-3, K)',
    )

    # Write propagated.
    for b_re, df in result.propagated.items():
        _write_daily_frames(
            df, output_dir,
            filename=f'IMF_{b_re}Re.dat',
            header_template=f'Propagated L1 Data for {{day}} (Target: {b_re} Re) (GSM nT, km/s, cm^-3, K)',
        )


def _write_daily_frames(df, output_dir, filename, header_template):
    """Split a DataFrame by day and write each as a .dat file."""
    if df.empty:
        return

    for date, group in df.groupby(df.index.date):
        day_str = date.strftime('%Y-%m-%d')
        day_dir = os.path.join(output_dir, date.strftime('%Y/%m/%d'))
        os.makedirs(day_dir, exist_ok=True)

        outpath = os.path.join(day_dir, filename)
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(header_template.format(day=day_str) + '\n')
            f.write('year mo dy hr mn Bx By Bz Ux Uy Uz rho T\n')
            f.write('#START\n')
            for t, row in group.iterrows():
                if pd.isna(row['Bx']):
                    continue
                f.write(
                    f"{t.year:4d} {t.month:2d} {t.day:2d} "
                    f"{t.hour:2d} {t.minute:2d} "
                    f"{row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} "
                    f"{row['Ux']:9.2f} {row['Uy']:9.2f} {row['Uz']:9.2f} "
                    f"{row['rho']:9.4f} {row['T']:10.1f}\n"
                )
    print(f'Wrote daily {filename} files to {output_dir}/')
