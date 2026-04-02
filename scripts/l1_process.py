# %%
import importlib
import logging
import os

import pandas as pd
import midlpy.l1_quality as l1_quality
import midlpy.l1_filters as l1_filters
import midlpy.l1_combine as l1_combine
import midlpy.l1_pipeline as l1_pipeline
from midlpy import create_combined_l1_files, process_day

# Set up logging to both console and a timestamped file.
log_file = f'l1_process_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# Force-reload pipeline modules so any code changes made this session
# are picked up without restarting the kernel.
for mod in [l1_filters, l1_quality, l1_pipeline, l1_combine]:
    importlib.reload(mod)
print('Modules reloaded.')

# %%

# Process from 2005-01-01 forward to 2024-12-31.
# Assumes L1_raw/ has already been populated by a download script.
start_date = pd.Timestamp('2005-01-01')
end_date = pd.Timestamp('2024-12-31')
days = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d').tolist()

start_time = pd.Timestamp.now()

# Seed: filter the day before start and day 0 so the combine step
# has prev_day context available from the first iteration.
day_before = (pd.Timestamp(days[0]) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
process_day(day_before)
process_day(days[0])

# Crawling window forwards: filter next_day, combine current day, advance.
for i, day in enumerate(days):
    try:
        prev_day = day_before if i == 0 else days[i - 1]

        if i < len(days) - 1:
            next_day = days[i + 1]
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
