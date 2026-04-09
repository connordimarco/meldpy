"""
l1_writers.py
-------------
Output formatters for MIDLResult objects.

Writes monthly CSV files to:
    output_dir/YYYY/MM/YYYYMM_{L1,14Re,32Re}.csv

Usage:
    from midlpy import midl, write_monthly_outputs

    result = midl('2024-05-09', '2024-05-11')
    write_monthly_outputs(result, output_dir='data')
"""
import os


def write_monthly_outputs(result, output_dir='data'):
    """Write MIDLResult to monthly CSV files.

    Creates directory structure:
        output_dir/YYYY/MM/YYYYMM_{L1,14Re,32Re}.csv

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
        df_unp['X'] = df_unp.index.map(
            lambda t: result.ref_x_re.get(t.date(), float('nan')))

    # Add source provenance columns.
    if result.source_map:
        for out_col, src_key in _SOURCE_COLUMNS.items():
            if src_key in result.source_map:
                src = result.source_map[src_key].reindex(df_unp.index)
                df_unp[out_col] = src.map(
                    lambda fs: _frozenset_to_str(fs) or float('nan'))

    datasets['L1'] = df_unp

    for b_re, df in result.propagated.items():
        datasets[f'{b_re}Re'] = df

    return datasets


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_CSV_PRECISION = {
    'Bx': '%.2f', 'By': '%.2f', 'Bz': '%.2f',
    'Ux': '%.1f', 'Uy': '%.2f', 'Uz': '%.2f',
    'rho': '%.3f', 'T': '%.0f', 'X': '%.1f',
}


def _write_csv(df, label, year_month_dir, ym_prefix):
    """Write one monthly CSV file."""
    os.makedirs(year_month_dir, exist_ok=True)
    path = os.path.join(year_month_dir, f'{ym_prefix}_{label}.csv')
    rounded = df.copy()
    for col, spec in _CSV_PRECISION.items():
        if col in rounded.columns:
            decimals = int(spec.split('.')[1].rstrip('f'))
            rounded[col] = rounded[col].round(decimals)
    rounded.to_csv(path, date_format='%Y-%m-%dT%H:%M:%S')
