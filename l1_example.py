# %%
from plot_l1_may2024 import plot_day
import l1_quality
import l1_filters
import l1_combine
import l1_pipeline
import importlib
from l1_combine import create_combined_l1_files
from l1_pipeline import get_one_day_swmf_input, create_position_file
from pyspedas import CDAWeb
import pandas as pd
import os
import sys
from pathlib import Path

# Locate the repo root as the directory containing this notebook.
# __vsc_ipynb_file__ is injected by VS Code; fall back to cwd for other environments.
try:
    NOTEBOOK_DIR = str(Path(__vsc_ipynb_file__).parent.resolve())
except NameError:
    NOTEBOOK_DIR = str(Path.cwd())

os.chdir(NOTEBOOK_DIR)
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

print(f"Working directory: {os.getcwd()}")

# %%

cda = CDAWeb()

# %%

# Force-reload pipeline modules so any code changes made this session
# are picked up without restarting the kernel.
for mod in [l1_filters, l1_quality, l1_pipeline, l1_combine]:
    importlib.reload(mod)
print('Modules reloaded.')

# %%
# -----------------------------------------------------------------------
# Date range to process.
# Uncomment the rolling-window block to always run the last N days.
# -----------------------------------------------------------------------

# today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
# days = pd.date_range(end=today, periods=7).strftime('%Y-%m-%d').tolist()


days = pd.date_range(
    start='2024-05-01', end='2024-05-31'
).strftime('%Y-%m-%d').tolist()

start_time = pd.Timestamp.now()

# Seed the window: download the day before day-1 (context only) and day-1.
day_before = (pd.Timestamp(days[0]) -
              pd.Timedelta(days=1)).strftime('%Y-%m-%d')
get_one_day_swmf_input(day_before, cda)
get_one_day_swmf_input(days[0], cda)
create_position_file(days[0], cda)

# Crawling window: download tomorrow, combine today (yesterday + tomorrow
# as context), plot, then advance.  Each day is downloaded exactly once.
for i, day in enumerate(days):
    prev_day = day_before if i == 0 else days[i - 1]

    if i < len(days) - 1:
        next_day = days[i + 1]
        get_one_day_swmf_input(next_day, cda)
        create_position_file(next_day, cda)
    else:
        next_day = None

    create_combined_l1_files(day, prev_day=prev_day, next_day=next_day)
    try:
        plot_day(day)
    except Exception as exc:
        print(f"  Plot error for {day}: {exc}")
    print(f"Completed: {day}")

end_time = pd.Timestamp.now()
print(f"\nDone. Elapsed: {end_time - start_time}")
