"""
midl_pipeline
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
write_monthly_outputs(result, output_dir)
    Write MIDLResult to monthly CSV and DAT files.
plot_day(result, day_str, output_dir)
    Plot all variables for one day.
plot_variable(result, var, day_str, output_dir)
    Plot a single variable for one day.
"""

from .l1_midl import midl, MIDLResult
from .l1_writers import write_monthly_outputs
from .l1_plot import plot_day, plot_variable, plot_day_from_csv
from .l1_pipeline import download_day
from .l1_mhd import mhd_propagation

__all__ = [
    'midl',
    'MIDLResult',
    'write_monthly_outputs',
    'plot_day',
    'plot_variable',
    'plot_day_from_csv',
    'download_day',
    'mhd_propagation',
]
