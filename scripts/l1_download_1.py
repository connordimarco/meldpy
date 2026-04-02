# %%
import logging
import os

import pandas as pd
from midlpy import download_day
from pyspedas import CDAWeb

# Set up logging to both console and a timestamped file.
log_file = f'l1_download_backward_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# Download from 2024-12-31 back to 2005-01-01 (back to front).
end_date = pd.Timestamp('2024-12-31')
start_date = pd.Timestamp('2005-01-01')
days = pd.date_range(start=start_date, end=end_date, freq='D')[::-1].strftime('%Y-%m-%d').tolist()

start_time = pd.Timestamp.now()

for day in days:
    try:
        download_day(day, cda)
        logger.info(f"Downloaded: {day}")
    except Exception as e:
        logger.error(f"Failed: {day} -- {e}", exc_info=True)
        continue

end_time = pd.Timestamp.now()
logger.info(f"\nDone. Elapsed: {end_time - start_time}")
