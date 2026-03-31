"""
l1_downloaders.py
-----------------
Download helpers for raw L1 data files.

Two sources are supported:
  - CDAWeb  (ACE, WIND, DSCOVR orbit) via the pyspedas CDAWeb client.
  - NOAA NGDC (DSCOVR 1-min plasma + mag) via direct HTTP requests.

Files are written to local scratch directories inside cdf_temp/ and are
expected to be cleaned up by the calling pipeline after processing.
"""
import os
import re
from datetime import datetime
import time

import requests


def download_cdaweb_files(cda, datasets, trange_start, trange_end, data_dir,
                          label='CDAWeb', max_attempts=3, retry_delay=30):
    """Download a set of CDAWeb datasets for a given time range.

    Parameters
    ----------
    cda : pyspedas.CDAWeb
        Authenticated CDAWeb client.
    datasets : list[str]
        Dataset IDs to request (e.g. ['AC_H0_MFI', 'WI_H0_MFI']).
    trange_start, trange_end : str
        ISO date strings 'YYYY-MM-DD' (or 'YYYY-MM-DD/HH:MM:SS').
    data_dir : str
        Local directory to write downloaded files into.
    label : str
        Human-readable label for log messages.
    max_attempts : int
        Maximum number of query/download attempts before giving up.
    retry_delay : int | float
        Seconds to wait between failed attempts.

    Returns
    -------
    urllist : list[str]
        List of URLs that were downloaded (empty if nothing found).
    """
    # Make sure the target folder exists before downloading.
    os.makedirs(data_dir, exist_ok=True)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            # Ask CDAWeb for files in the requested time window.
            print(f'Querying {label} (attempt {attempt}/{max_attempts})...')
            urllist = cda.get_filenames(datasets, trange_start, trange_end)

            if not urllist:
                # Nothing found is okay; caller can decide what to do.
                print('No files found.')
                return []

            # Pull files to local scratch storage.
            print(f'Downloading {len(urllist)} files...')
            cda.cda_download(urllist, local_dir=data_dir, download_only=True)
            return urllist
        except Exception as exc:
            last_error = exc
            print(f'WARNING: {label} download attempt {attempt} failed: {exc}')
            if attempt < max_attempts:
                print(f'Waiting {retry_delay} seconds before retrying...')
                time.sleep(retry_delay)

    raise RuntimeError(
        f'Failed to download {label} after {max_attempts} attempts: {last_error}'
    ) from last_error


def download_position_cdaweb_files(cda, day, data_dir,
                                   max_attempts=3, retry_delay=30):
    """Download a narrow (~2 h) noon window of position data for one day.

    Fetches ACE MFI, WIND MFI, and DSCOVR orbit CDF files.
    Only 11:00–13:00 UT is requested because a stable mean position is all
    that downstream propagation needs.

    Parameters
    ----------
    cda : pyspedas.CDAWeb
    day : str  ('YYYY-MM-DD')
    data_dir : str
    max_attempts : int
    retry_delay : int | float

    Returns
    -------
    urllist : list[str]
    """
    # Position file generation only needs a 2-hour noon window.
    dt_day = datetime.strptime(day, '%Y-%m-%d')
    trange_start = (dt_day.replace(hour=11, minute=0, second=0)
                    ).strftime('%Y-%m-%d/%H:%M:%S')
    trange_end = (dt_day.replace(hour=13, minute=0, second=0)
                  ).strftime('%Y-%m-%d/%H:%M:%S')
    datasets = ['AC_H0_MFI', 'WI_H0_MFI', 'DSCOVR_ORBIT_PRE']

    return download_cdaweb_files(
        cda,
        datasets,
        trange_start,
        trange_end,
        data_dir,
        label='Position Data (11:00-13:00 UT)',
        max_attempts=max_attempts,
        retry_delay=retry_delay,
    )


def download_dscovr_ngdc(day, data_dir, products=('f1m', 'm1m')):
    """Download DSCOVR 1-minute products from the NOAA NGDC portal.

    Fetches gzipped NetCDF files matching the given date from
    https://www.ngdc.noaa.gov/dscovr/data/YYYY/MM/.

    Parameters
    ----------
    day : str  ('YYYY-MM-DD')
    data_dir : str
    products : tuple[str]
        NGDC product codes to fetch.  Defaults: 'f1m' (plasma), 'm1m' (mag).

    Returns
    -------
    paths : dict[str, str]
        Maps product code -> local file path for each successfully downloaded file.

    Raises
    ------
    RuntimeError
        If the NGDC monthly directory listing cannot be fetched.
    """
    # Build date-specific directory/filename pieces once.
    dt = datetime.strptime(day, '%Y-%m-%d')
    date_str = dt.strftime('%Y%m%d')
    base_url = f"https://www.ngdc.noaa.gov/dscovr/data/{dt.year}/{dt.month:02d}/"

    os.makedirs(data_dir, exist_ok=True)

    # Grab the monthly index page and search for matching product files.
    try:
        resp = requests.get(base_url, timeout=30)
        resp.raise_for_status()
        listing = resp.text
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch NGDC directory {base_url}: {e}") from e

    paths = {}
    for product in products:
        # NGDC filenames include run-specific tags, so we match by regex.
        pattern = rf'oe_{re.escape(product)}_dscovr_s{date_str}\d+_e{date_str}\d+_p\d+_pub\.nc\.gz'
        match = re.search(pattern, listing)
        if not match:
            print(f"  WARNING: No NGDC {product} file found for {day}.")
            continue

        filename = match.group(0)
        file_url = base_url + filename
        local_path = os.path.join(
            data_dir, f"dscovr_{product}_{date_str}.nc.gz")

        try:
            # Stream download to avoid loading the whole file into memory.
            r = requests.get(file_url, timeout=120, stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            paths[product] = local_path
            print(f"  Downloaded {filename} -> {local_path}")
        except Exception as e:
            print(f"  WARNING: Failed to download {file_url}: {e}")

    return paths
