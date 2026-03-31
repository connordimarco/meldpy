# %%
import importlib
import logging
import os

import pandas as pd
import meldpy.l1_quality as l1_quality
import meldpy.l1_filters as l1_filters
import meldpy.l1_combine as l1_combine
import meldpy.l1_pipeline as l1_pipeline
from meldpy import create_combined_l1_files, download_day, process_day
from pyspedas import CDAWeb

# Set up logging to both console and a timestamped file.
log_file = f'l1_backward_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Logging to: {log_file}")

# %%

cda = CDAWeb()

# %%

# Force-reload pipeline modules so any code changes made this session
# are picked up without restarting the kernel.
for mod in [l1_filters, l1_quality, l1_pipeline, l1_combine]:
    importlib.reload(mod)
print('Modules reloaded.')

# %%

# Start from 2024-12-31 and work backwards to 2005-01-01.
# Cap at 2024-12-31 -- SpacePy IGRF model valid range ends 2025-01-01.
end_date = pd.Timestamp('2024-12-31')
start_date = pd.Timestamp('2005-01-01')
days = pd.date_range(start=start_date, end=end_date, freq='D')[::-1].strftime('%Y-%m-%d').tolist()

start_time = pd.Timestamp.now()

# Seed: download tomorrow (next_day context for today) and today.
day_after = (pd.Timestamp(days[0]) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
download_day(day_after, cda)
process_day(day_after)
download_day(days[0], cda)
process_day(days[0])

# Crawling window backwards: download prev_day, combine current day, advance.
# Each day is downloaded exactly once.
for i, day in enumerate(days):
    try:
        next_day = day_after if i == 0 else days[i - 1]

        if i < len(days) - 1:
            prev_day = days[i + 1]
            download_day(prev_day, cda)
            process_day(prev_day)
        else:
            prev_day = None

        create_combined_l1_files(day, prev_day=prev_day, next_day=next_day)
        logger.info(f"Completed: {day}")
    except Exception as e:
        logger.error(f"Failed: {day} -- {e}", exc_info=True)
        continue

end_time = pd.Timestamp.now()
logger.info(f"\nDone. Elapsed: {end_time - start_time}")
