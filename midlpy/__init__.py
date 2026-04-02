"""
midlpy
------
Multi-satellite solar wind merging pipeline for SWMF/BATS-R-US boundary conditions.

Merges 1-minute data from ACE, DSCOVR, and WIND into quality-screened upstream
boundary condition files.

Public API
----------
midl(start, end, raw_dir)
    Process a date range: read L1_raw, despike, quality-score, merge, propagate.
    Returns a MIDLResult with unpropagated and propagated DataFrames.
download_day(day, cda, raw_dir)
    Download raw satellite data to raw_dir/.
write_monthly_parquet(result, output_dir)
    Write MIDLResult to monthly Parquet files.
write_daily_dat(result, output_dir)
    Write MIDLResult to per-day .dat files.
plot_day(result, day_str, output_dir)
    Plot all variables for one day.
plot_variable(result, var, day_str, output_dir)
    Plot a single variable for one day.
"""

from .l1_midl import midl, MIDLResult
from .l1_writers import write_monthly_parquet, write_daily_dat
from .l1_plot import plot_day, plot_variable, plot_day_from_parquet
from .l1_pipeline import download_day, process_day, get_one_day_swmf_input
from .l1_combine import create_combined_l1_files

__all__ = [
    'midl',
    'MIDLResult',
    'write_monthly_parquet',
    'write_daily_dat',
    'plot_day',
    'plot_variable',
    'plot_day_from_parquet',
    'download_day',
    'process_day',
    'create_combined_l1_files',
    'get_one_day_swmf_input',
]
