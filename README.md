# Multi-Satellite Integrated Dataset from L1 (MIDL)

Downloads, quality-screens, and combines 1-minute solar wind from **ACE**, **DSCOVR**, and **WIND** into merged time series.

---

## Data Flow

```mermaid
flowchart TD
    subgraph PIPE["l1_pipeline.py — Per-Satellite Processing"]
        ACE["ACE · CDAWeb"]
        DSC["DSCOVR · NOAA NGDC"]
        WND["WIND · CDAWeb"]
        DL["Download raw CDF / NetCDF\nl1_downloaders.py"]
        RD["Read and parse raw files\nl1_readers.py"]
        RS["Resample to 1-min grid\nGSE→GSM rotation · unit conversion"]
        DS1["Despike: 3-point median filter\nl1_filters.py"]
        ACE & DSC & WND --> DL --> RD --> RS --> DS1
    end

    L1R[/"L1_raw / YYYY/MM/DD /\nper-satellite .dat files"/]

    RS  --> L1R

    subgraph COMB["l1_midl.py — Merge Pipeline"]
        LD["Load per-satellite .dat files\n+1 day padding for context"]
        GF["Gap fill per satellite\nB ≤ 5 min · plasma ≤ 60 min"]
        PR["Propagate to reference position\nballistic shift to nearest satellite"]

        subgraph QC["Plasma Quality — l1_quality.py"]
            Q1["Outlier detection\n2-of-3 agreement check"]
            Q3["NaN-fraction check\n60-min rolling window"]
            Q4["Flat-plateau detection\nstuck instrument readings"]
            Q5["Near-zero Uy/Uz\nDSCOVR Faraday cup artifact"]
        end

        VS["Source selection\nB: coupled via magnitude\nUy/Uz: coupled via transverse speed\nUx, rho: independent"]

        subgraph TC["Temperature — combine_temperature"]
            T1["Per-satellite 3-pt median"]
            T2["Log-std spikiness filter"]
            T3["Geometric median\nacross satellites"]
            T4["Final 3-pt median smooth"]
            T1 --> T2 --> T3 --> T4
        end

        ST["Smooth source transitions\nl1_filters.py"]
        SL["Slice to requested range\nfill residual NaN gaps"]

        LD  --> GF
        GF  --> PR
        PR  --> QC
        PR  --> T1
        Q1 & Q3 & Q4 & Q5 --> VS
        VS  --> ST
        T4  --> ST
        ST  --> SL
    end

    OUT[/"Monthly output\ndata / YYYY/MM / csv + dat"/]

    subgraph PROP["Ballistic Propagation — l1_propagation.py"]
        BP["Propagate to boundary\nΔt = ΔX / Ux\ncausality enforced"]
    end

    IMF14[/"Propagated to 14 Re"/]
    IMF32[/"Propagated to 32 Re"/]

    L1R --> LD
    SL  --> OUT
    SL  --> PROP
    BP  --> IMF14
    BP  --> IMF32
```
---

## File Inventory

| File | Role |
|---|---|
| `l1_midl.py` | **Primary entry point**: `midl(start, end)` continuous pipeline, returns `MIDLResult` |
| `l1_writers.py` | Output formatters: `write_monthly_outputs()` (CSV + DAT) |
| `l1_plot.py` | Debugging plots: `plot_day()`, `plot_variable()` |
| `l1_pipeline.py` | Download, resample, coordinate rotation, and per-satellite raw `.dat` output |
| `l1_combine.py` | Source selection, satellite merging, and temperature combining logic |
| `l1_quality.py` | Quality checks and `score_all_plasma()` |
| `l1_filters.py` | `despike()`, `smooth_transitions()`, `median_filter_3()`, `interpolate_with_limits()` |
| `l1_propagation.py` | Ballistic travel-time propagation with causality enforcement |
| `l1_readers.py` | CDF and gzipped NetCDF readers; ASCII `.dat` reader |
| `l1_downloaders.py` | CDAWeb and NOAA NGDC download helpers |

---

## Output Layout

### Raw per-satellite output

`L1_raw/YYYY/MM/DD/`

| File | Description |
|---|---|
| `L1_ace.dat` | ACE 1-min stream before filtering |
| `L1_dscovr.dat` | DSCOVR 1-min stream before filtering |
| `L1_wind.dat` | WIND 1-min stream before filtering |

### Monthly pipeline output

`data/YYYY/MM/{csv,dat}/`

| File | Description |
|---|---|
| `YYYYMM_unpropagated.{csv,dat}` | Merged stream at reference satellite position. Includes `X_Re` and source provenance columns (`B_source`, `Ux_source`, `Uyz_source`, `rho_source`, `T_source`). Source values are satellite codes: 1=ACE, 2=DSCOVR, 3=WIND, concatenated (e.g. `13` = ACE+WIND). |
| `YYYYMM_14Re.{csv,dat}` | Combined stream propagated to 14 Re |
| `YYYYMM_32Re.{csv,dat}` | Combined stream propagated to 32 Re |

Column layout is compatible with SWMF/BATS-R-US upstream input readers.

---

## Data Sources

| Satellite | Magnetometer | Plasma | Source |
|---|---|---|---|
| ACE | `AC_H0_MFI` (GSM) | `AC_H0_SWE` (GSM) | CDAWeb |
| DSCOVR | NGDC `m1m` (GSE → GSM) | NGDC `f1m` (GSE → GSM) | NOAA NGDC |
| WIND | `WI_H0_MFI` (GSM) | `WI_H1_SWE` (GSE → GSM) | CDAWeb |

DSCOVR plasma is taken from NOAA NGDC because the CDAWeb Faraday cup plasma product ends around 2019.

> **Note:** GSE→GSM coordinate rotation uses spacepy's IGRF geomagnetic field model, which has a finite validity window (currently through 2030 with IGRF14). Processing dates beyond this range requires updating spacepy to a version with newer IGRF coefficients.

---

## Methodology

Full algorithm description in the accompanying manuscript. For tunable parameters, see below.

---

## Tunable Parameters

- Quality thresholds: module-level constants in `l1_quality.py`
- Agreement thresholds (when satellites "agree"): `_switch_threshold()` in `l1_combine.py`
- Fallback hysteresis: `_SWITCH_MIN = 3` in `_select_column_with_continuity()` in `l1_combine.py`
- Transition smoothing: `_CMAX_DEFAULT`, `_WMAX_DEFAULT`, `_RATE_DEFAULT` in `l1_filters.py`
- Filter behavior: `despike()` in `l1_filters.py`
- Temperature combiner: `combine_temperature()` in `l1_combine.py`
