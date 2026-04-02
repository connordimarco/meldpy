# %%
"""
l1_reprocess.py
---------------
Reprocesses the full L1 dataset using the midl() pipeline.

Run from the Propagation/ working directory:
    S:/conda/python.exe midlpy/scripts/l1_reprocess.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from midlpy import midl

# Import sibling scripts (not part of the library).
sys.path.insert(0, os.path.dirname(__file__))
from l1_writers import write_monthly_parquet, write_daily_dat
from l1_plot import plot_day

# %%
# --- Process a test range first ---
result = midl('2024-05-09', '2024-05-11')

# Quick visual check.
plot_day(result, '2024-05-10')

# %%
# --- Write outputs ---
write_daily_dat(result, output_dir='L1')
write_monthly_parquet(result, output_dir='L1_db/data')

# %%
# --- Full 20-year reprocess (run on cluster) ---
# Process year-by-year to manage memory.
#
# for year in range(2005, 2025):
#     print(f'\n=== Processing {year} ===')
#     result = midl(f'{year}-01-01', f'{year}-12-31')
#     write_monthly_parquet(result, output_dir='L1_db/data')
#     del result
