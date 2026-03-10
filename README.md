# L1 Solar Wind Pipeline

Downloads, quality-screens, and combines 1-minute solar wind from **ACE**, **DSCOVR**, and **WIND** into a merged time series for SWMF/BATS-R-US upstream boundary conditions.

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
| `plot_l1_may2024.py` | Diagnostic 4-column multi-panel plots |
| `l1_example.ipynb` | Driver notebook — end-to-end example for a date range |

---

## Output files

All outputs land in `L1/YYYY/MM/DD/`.

| File | Description |
|---|---|
| `L1_ace.dat` | ACE 1-min data (GSM) |
| `L1_dscovr.dat` | DSCOVR 1-min data (GSM) |
| `L1_wind.dat` | WIND 1-min data (GSM) |
| `L1_satpos.dat` | Noon GSM positions (Re) for all three satellites |
| `L1_combined.dat` | Merged, quality-screened, despiked time series + nSat/satUsed columns |
| `IMF_14Re.dat` | Combined data ballistically propagated to 14 Re |
| `IMF_32Re.dat` | Combined data ballistically propagated to 32 Re |

Column layout is compatible with SWMF/BATS-R-US upstream input readers.

---

## Data sources

| Satellite | Magnetometer | Plasma | Source |
|---|---|---|---|
| ACE | AC_H0_MFI (GSM) | AC_H0_SWE (GSM) | CDAWeb |
| DSCOVR | NGDC m1m (GSM) | NGDC f1m (GSM) | NOAA NGDC |
| WIND | WI_H0_MFI (GSM) | WI_H1_SWE (GSE→GSM) | CDAWeb |

DSCOVR plasma comes from [NOAA NGDC](https://www.ngdc.noaa.gov/dscovr/portal/) because the CDAWeb Faraday cup data ends ~2019. WIND plasma vectors are rotated GSE→GSM; ACE and DSCOVR mag are already GSM at source.

---

## Quality checks

Each satellite/variable/minute is tested by six independent checks. Anything flagged bad is removed from the candidate pool before combination.

1. **Outlier detection** (all satellites) — When two satellites agree and the third doesn't, the odd one out is flagged. Both must agree *with each other* for the flag to fire, so a single bad satellite can't poison the reference.

2. **DSCOVR pairwise check** — Compares DSCOVR against ACE/WIND directly. Catches DSCOVR issues when only two satellites have data and the 3-satellite outlier logic can't run. Uses a tighter 2× threshold for rho/T (vs. 3× in the outlier check) because DSCOVR can sit 1.5–2× high for hours.

3. **Physical range** (all satellites) — Throws out values outside plausible bounds (e.g. Ux must be Earthward, rho 0.1–100 cm⁻³, T up to 5×10⁶ K).

4. **NaN fraction** (all satellites) — If more than half of a 60-min window is NaN, the remaining points in that window are treated as unreliable too.

5. **Flat-plateau detection** (DSCOVR only) — Catches stuck or near-constant readings using rolling standard deviation and unique-value counts. The DSCOVR Faraday cup sometimes returns repeating values that look valid but aren't.

6. **Near-zero Uy/Uz** (DSCOVR only) — Flags transverse velocity within ±0.5 km/s of zero (DSCOVR occasionally zeroes these out).

---

## Source selection

After quality gating, the best source is picked each minute:

| Satellites passing QC | Rule |
|---|---|
| All three | Median |
| WIND + one other | Use WIND |
| ACE + DSCOVR — density | Use DSCOVR |
| ACE + DSCOVR — velocity, temperature | Use ACE |
| One only | Use that satellite |
| None | NaN → interpolated if gap ≤ 20 min |

A continuity guard prevents large jumps when the source switches — if the preferred satellite would be too far from the previous value, the closest available satellite is used instead.

The combined file includes `nSat` (how many satellites passed QC) and `satUsed` (1 = ACE, 2 = DSCOVR, 3 = WIND).

---

## Tunable parameters

Quality thresholds live as module-level constants in `l1_quality.py`:

```python
_PLATEAU_PARAMS   # flat-plateau: window, std_thresh, max_unique per variable
_INTERSC_PARAMS   # DSCOVR pairwise check: mode, threshold, window
_OUTLIER_PARAMS   # symmetric outlier check: mode, threshold, window
_PHYSICAL_BOUNDS  # range checks: min/max per variable
```

Continuity-guard thresholds are in `_switch_threshold()` in `l1_combine.py`. Despike parameters are in `despike()` in `l1_filters.py`.

