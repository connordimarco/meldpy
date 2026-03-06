# L1 Solar Wind Pipeline

Automated pipeline that downloads, quality-screens, and combines 1-minute solar wind observations from three L1 spacecraft — **ACE**, **DSCOVR**, and **WIND** — into a single merged time series suitable for use as upstream boundary conditions in SWMF/BATS-R-US simulations.

---

## Overview

Raw data from each satellite is downloaded, resampled to a common 1-minute grid, and written to per-satellite `.dat` files. A quality scorer then evaluates every satellite/variable/minute before combination, so bad observations are excluded before a source is selected. The combined file is ballistically propagated to inner boundaries (14 Re and 32 Re by default) and diagnostic plots are generated.

The pipeline processes days in a **crawling window** — each day is downloaded exactly once, and quality-scoring and despike filters run over a 3-day window (yesterday + today + tomorrow) so rolling statistics are fully warmed up at day boundaries. Only today's slice is written to output files.

---

## Data sources

| Satellite | Magnetometer | Plasma | Source |
|---|---|---|---|
| ACE | AC_H0_MFI (GSM, nT) | AC_H0_SWE (GSM, km/s, cm⁻³, K) | CDAWeb |
| DSCOVR | NGDC m1m (GSM, nT) | NGDC f1m (GSM, km/s, cm⁻³, K) | NOAA NGDC |
| WIND | WI_H0_MFI (GSM, nT) | WI_H1_SWE (GSE→GSM, km/s, cm⁻³, K) | CDAWeb |

DSCOVR plasma is sourced from [NOAA NGDC](https://www.ngdc.noaa.gov/dscovr/portal/) rather than CDAWeb because the CDAWeb Faraday cup data ends ~2019. DSCOVR magnetometer data is pre-rotated to GSM at source. WIND and DSCOVR plasma vectors are rotated GSE→GSM before output; ACE is already in GSM.

---

## Quality assessment

Before combination, every satellite/variable/minute passes through five independent checks. A minute flagged bad by any check is removed from the candidate pool for that variable.

### 1. Outlier detection (all three satellites)
All pairwise deviations are computed (ACE–DSCOVR, ACE–WIND, DSCOVR–WIND) over a rolling median window. A satellite is only flagged when the other two **agree with each other** and it disagrees with both — this prevents a bad satellite from contaminating the reference used to judge the others.

### 2. Pairwise DSCOVR consistency check
A dedicated ACE/WIND-vs-DSCOVR comparison catches DSCOVR issues in 2-satellite scenarios (e.g. during WIND data gaps) when the 3-satellite outlier logic cannot fire.

Comparisons are absolute for velocity and log-ratio for density/temperature so that a 3× over- and 3× underestimate are treated symmetrically.

| Variable | Mode | Threshold | Window |
|---|---|---|---|
| Ux | absolute | 50 km/s | 31 min |
| Uy, Uz | absolute | 30 km/s | 31 min |
| rho | log-ratio | **2×** | 61 min |
| T | log-ratio | **2×** | 31 min |

The rho/T threshold here (2×) is intentionally tighter than the outlier check above (3×). DSCOVR can be persistently elevated by a moderate 1.5–2× for hours at a time — enough to corrupt the merged output but not enough to trigger a 3× threshold. The outlier check is kept at 3× because applying 2× symmetrically causes ACE to be incorrectly flagged on days where DSCOVR and WIND are both elevated and happen to "agree" with each other.

### 3. Physical range checks (all satellites)

| Variable | Min | Max |
|---|---|---|
| Ux | −2500 km/s | −150 km/s (must be Earthward) |
| Uy, Uz | −200 km/s | +200 km/s |
| rho | 0.1 cm⁻³ | 100 cm⁻³ |
| T | 1 000 K | 5 × 10⁶ K |

### 4. NaN-fraction / data-gap metric (all satellites)
If more than 50 % of a 60-minute rolling window is NaN, the surrounding non-NaN points in the same window are also considered unreliable.

### 5. Flat-plateau detection (DSCOVR only)
Detects stuck or near-constant instrument readings via a centered rolling standard deviation and unique-value count. The DSCOVR Faraday cup is known to occasionally return repeating values that appear valid but are not physical.

| Variable | Window | Std threshold |
|---|---|---|
| Ux | 15 min | 1.0 km/s |
| Uy, Uz | 11 min | 0.08 km/s |
| rho | 15 min | 0.05 cm⁻³ |
| T | 15 min | 500 K |

### 6. Near-zero Uy/Uz (DSCOVR only)
Flags Uy or Uz within ±0.5 km/s of zero as suspect (DSCOVR occasionally zeroes out transverse velocity components).

---

## Source selection hierarchy

After quality gating, the best available source is chosen for each variable/minute:

| Satellites passing QC | Rule |
|---|---|
| All three | Median |
| WIND + one other | Use WIND |
| ACE + DSCOVR — density | Use DSCOVR |
| ACE + DSCOVR — velocity, temperature | Use ACE |
| One only | Use that satellite |
| None | NaN → interpolated if gap ≤ 20 min |

A **continuity guard** prevents large jumps when the source changes: if the preferred candidate's value differs from the previous output by more than a per-variable threshold, whichever available satellite is closest to the previous value is chosen instead.

The combined file includes two extra provenance columns: `nSat` (number of satellites that passed QC this minute) and `satUsed` (which was chosen: 1 = ACE, 2 = DSCOVR, 3 = WIND).

---

## Output files

All outputs land in `L1/YYYY/MM/DD/` relative to this directory.

| File | Description |
|---|---|
| `L1_ace.dat` | Quality-unfiltered ACE 1-min data (GSM, nT / km/s / cm⁻³ / K) |
| `L1_dscovr.dat` | Quality-unfiltered DSCOVR 1-min data (GSM) |
| `L1_wind.dat` | Quality-unfiltered WIND 1-min data (GSM) |
| `L1_satpos.dat` | Noon GSM positions in Re for all three satellites |
| `L1_combined.dat` | Merged, quality-screened, despiked time series + nSat/satUsed columns |
| `IMF_14Re.dat` | `L1_combined.dat` ballistically propagated to 14 Re |
| `IMF_32Re.dat` | `L1_combined.dat` ballistically propagated to 32 Re |

The column layout of all files is compatible with SWMF/BATS-R-US upstream input readers.

---

## Running the pipeline

`l1_example.ipynb` is the driver notebook. Run cells in order:

**Cell 1** — sets the working directory so all `l1_*` module imports resolve correctly. Edit `_PATHS` if running on a new machine.

**Cell 2** — imports and initialises the CDAWeb client.

**Cell 3** — main processing loop. Adjust `start` / `end` in `pd.date_range()` to set the date range, then run. Each day is downloaded exactly once via the crawling window:
1. Seed: download the day before day-1 and day-1 itself.
2. Loop: download tomorrow → combine today (prev + today + next context) → plot today → advance.

**Cell 4 (standalone)** — generates context plots: each day shown with ±6 h of the neighbouring days. Dashed vertical lines mark midnight boundaries so cross-day continuity is immediately visible. Outputs go to `plots/with_context/`. Can be run without running Cell 3 first.

---

## Tunable parameters

All quality thresholds are module-level constants in `l1_quality.py` and can be adjusted without touching any logic:

```python
_PLATEAU_PARAMS   # flat-plateau: window, std_thresh, max_unique per variable
_INTERSC_PARAMS   # DSCOVR pairwise check: mode, threshold (2× for rho/T), window
_OUTLIER_PARAMS   # symmetric outlier check: mode, threshold (3× for rho/T), window
_PHYSICAL_BOUNDS  # range checks: min/max per variable

# check_nan_fraction():  window (default 60 min), threshold (default 0.5)
# check_near_zero():     atol   (default 0.5 km/s)
```

Continuity-guard jump thresholds are in `_switch_threshold()` in `l1_combine.py`.

Despike filter parameters (`limit_growth` factor and `limit_change` bounds) are in `despike()` in `l1_filters.py`.

---

## File inventory

| File | Role |
|---|---|
| `l1_pipeline.py` | Download, resample, coordinate rotation, per-satellite `.dat` output |
| `l1_combine.py` | Multi-satellite merge with quality gating, source selection, despike, propagation |
| `l1_quality.py` | All quality checks and the `score_all_plasma()` entry point |
| `l1_filters.py` | `limit_growth()`, `limit_change()`, `despike()` |
| `l1_propagation.py` | Ballistic travel-time propagation with causality enforcement |
| `l1_readers.py` | CDF and gzipped NetCDF readers; ASCII `.dat` reader |
| `l1_downloaders.py` | CDAWeb and NOAA NGDC download helpers |
| `l1_coordinates.py` | GSE → GSM rotation via SpacePy |
| `plot_l1_may2024.py` | Diagnostic 4-column multi-panel plots (`plot_day`, `plot_day_with_context`) |
| `l1_example.ipynb` | Driver notebook — end-to-end example for a date range |

