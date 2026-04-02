"""
midlpy
------
Multi-satellite solar wind merging pipeline for SWMF/BATS-R-US boundary conditions.

Merges 1-minute data from ACE, DSCOVR, and WIND into quality-screened upstream
boundary condition files.

Public API
----------
midl(start, end)
    Process a date range: read L1_raw, despike, quality-score, merge, propagate.
    Returns a MIDLResult with unpropagated and propagated DataFrames.
download_day(day, cda)
    Download raw satellite data to L1_raw/.

Output directories (L1/, L1_raw/) are created relative to the current working
directory -- run from wherever you want the data to land.
"""

from .l1_midl import midl, MIDLResult
from .l1_pipeline import download_day, process_day, get_one_day_swmf_input
from .l1_combine import create_combined_l1_files

__all__ = [
    'midl',
    'MIDLResult',
    'download_day',
    'process_day',
    'create_combined_l1_files',
    'get_one_day_swmf_input',
]
