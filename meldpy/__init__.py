"""
meldpy
------
Multi-satellite solar wind merging pipeline for SWMF/BATS-R-US boundary conditions.

Merges 1-minute data from ACE, DSCOVR, and WIND into quality-screened upstream
boundary condition files (IMF_14Re.dat, IMF_32Re.dat).

Public API
----------
download_day(day, cda)
    Phase 1: download raw satellite data to L1_raw/.
process_day(day)
    Phase 2: despike, filter, write per-satellite files to L1/.
create_combined_l1_files(day, prev_day, next_day)
    Phase 3: merge satellites, quality-gate, propagate to 14/32 Re.

Output directories (L1/, L1_raw/) are created relative to the current working
directory -- run from wherever you want the data to land.
"""

from .l1_pipeline import download_day, process_day, get_one_day_swmf_input
from .l1_combine import create_combined_l1_files

__all__ = [
    'download_day',
    'process_day',
    'create_combined_l1_files',
    'get_one_day_swmf_input',
]
