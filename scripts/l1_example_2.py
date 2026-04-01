# %%
import importlib
import logging
import os

import pandas as pd
import midlpy.l1_quality as l1_quality
import midlpy.l1_filters as l1_filters
import midlpy.l1_combine as l1_combine
import midlpy.l1_pipeline as l1_pipeline
from midlpy import create_combined_l1_files, download_day, process_day
from pyspedas import CDAWeb

# Set up logging to both console and a timestamped file.
log_file = f'l1_forward_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# Start from 2005-01-01 and count forward to 2024-12-31.
# Cap at 2024-12-31 -- SpacePy IGRF model valid range ends 2025-01-01.
start_date = pd.Timestamp('2005-01-01')
end_date = pd.Timestamp('2024-12-31')
days = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d').tolist()

start_time = pd.Timestamp.now()

# Seed: download the day before start (prev_day context for day 0) and day 0.
day_before = (pd.Timestamp(days[0]) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
download_day(day_before, cda)
process_day(day_before)
download_day(days[0], cda)
process_day(days[0])

# Crawling window forwards: download next_day, combine current day, advance.
# Each day is downloaded exactly once.
for i, day in enumerate(days):
    try:
        prev_day = day_before if i == 0 else days[i - 1]

        if i < len(days) - 1:
            next_day = days[i + 1]
            download_day(next_day, cda)
            process_day(next_day)
        else:
            next_day = None

        create_combined_l1_files(day, prev_day=prev_day, next_day=next_day)
        logger.info(f"Completed: {day}")
    except Exception as e:
        logger.error(f"Failed: {day} -- {e}", exc_info=True)
        continue

end_time = pd.Timestamp.now()
logger.info(f"\nDone. Elapsed: {end_time - start_time}")
