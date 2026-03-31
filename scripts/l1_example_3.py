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

log_file = f'l1_middle_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
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

for mod in [l1_filters, l1_quality, l1_pipeline, l1_combine]:
    importlib.reload(mod)
print('Modules reloaded.')

# %%

# Spiral outward from 2015-01-01 toward both ends of the 20-year range.
CENTER = pd.Timestamp('2015-01-01')
FORWARD_LIMIT = pd.Timestamp('2024-12-31')
BACKWARD_LIMIT = pd.Timestamp('2005-01-01')
ONE_DAY = pd.Timedelta(days=1)


def day_str(ts):
    return ts.strftime('%Y-%m-%d')


def fetch(day_ts):
    """Download + Stage 1 process a single day."""
    d = day_str(day_ts)
    download_day(d, cda)
    process_day(d)


def combine(day_ts, prev_ts, next_ts):
    """Stage 2 combine for a single day."""
    d = day_str(day_ts)
    p = day_str(prev_ts) if prev_ts is not None else None
    n = day_str(next_ts) if next_ts is not None else None
    create_combined_l1_files(d, prev_day=p, next_day=n)


start_time = pd.Timestamp.now()

# --- Seed: download + process center and its two neighbors ---
logger.info(f"Seeding center: {day_str(CENTER)}")
for d in [CENTER - ONE_DAY, CENTER, CENTER + ONE_DAY]:
    fetch(d)

combine(CENTER, CENTER - ONE_DAY, CENTER + ONE_DAY)
logger.info(f"Completed: {day_str(CENTER)}")

# --- Spiral outward, alternating forward and backward ---
fwd = CENTER + ONE_DAY       # next day to combine going forward
bwd = CENTER - ONE_DAY       # next day to combine going backward
fwd_edge = CENTER + ONE_DAY  # furthest day downloaded+processed forward
bwd_edge = CENTER - ONE_DAY  # furthest day downloaded+processed backward

while fwd <= FORWARD_LIMIT or bwd >= BACKWARD_LIMIT:
    # Forward step
    if fwd <= FORWARD_LIMIT:
        try:
            # Download the next context day if we haven't yet.
            need = fwd + ONE_DAY
            if need > fwd_edge:
                fetch(need)
                fwd_edge = need
            combine(fwd, fwd - ONE_DAY, fwd + ONE_DAY)
            logger.info(f"Completed (fwd): {day_str(fwd)}")
        except Exception as e:
            logger.error(f"Failed (fwd): {day_str(fwd)} -- {e}", exc_info=True)
        fwd += ONE_DAY

    # Backward step
    if bwd >= BACKWARD_LIMIT:
        try:
            need = bwd - ONE_DAY
            if need < bwd_edge:
                fetch(need)
                bwd_edge = need
            combine(bwd, bwd - ONE_DAY, bwd + ONE_DAY)
            logger.info(f"Completed (bwd): {day_str(bwd)}")
        except Exception as e:
            logger.error(f"Failed (bwd): {day_str(bwd)} -- {e}", exc_info=True)
        bwd -= ONE_DAY

end_time = pd.Timestamp.now()
logger.info(f"\nDone. Elapsed: {end_time - start_time}")
