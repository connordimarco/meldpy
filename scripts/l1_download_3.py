# %%
import logging
import os

import pandas as pd
from midlpy import download_day
from pyspedas import CDAWeb

# Set up logging to both console and a timestamped file.
log_file = f'l1_download_middle_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# Spiral outward from 2015-01-01 toward both ends of the 20-year range.
CENTER = pd.Timestamp('2015-01-01')
FORWARD_LIMIT = pd.Timestamp('2024-12-31')
BACKWARD_LIMIT = pd.Timestamp('2005-01-01')
ONE_DAY = pd.Timedelta(days=1)

start_time = pd.Timestamp.now()

fwd = CENTER
bwd = CENTER - ONE_DAY

while fwd <= FORWARD_LIMIT or bwd >= BACKWARD_LIMIT:
    if fwd <= FORWARD_LIMIT:
        try:
            download_day(fwd.strftime('%Y-%m-%d'), cda)
            logger.info(f"Downloaded (fwd): {fwd.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"Failed (fwd): {fwd.strftime('%Y-%m-%d')} -- {e}", exc_info=True)
        fwd += ONE_DAY

    if bwd >= BACKWARD_LIMIT:
        try:
            download_day(bwd.strftime('%Y-%m-%d'), cda)
            logger.info(f"Downloaded (bwd): {bwd.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"Failed (bwd): {bwd.strftime('%Y-%m-%d')} -- {e}", exc_info=True)
        bwd -= ONE_DAY

end_time = pd.Timestamp.now()
logger.info(f"\nDone. Elapsed: {end_time - start_time}")
