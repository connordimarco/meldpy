"""
plot_l1_may2024.py
------------------
For each day in May 2024, reads the L1 .dat files and produces a multi-panel
figure with:
  Rows    : each physical variable (Bx, By, Bz, Ux, Uy, Uz, rho, T)
  Columns : (1) raw satellites | (2) filtered satellites | (3) combined |
            (4) combined + propagated (14 Re, 32 Re)

Output is saved to plots/YYYY_MM_DD.png.
"""

from datetime import datetime
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from l1_readers import read_l1_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COL_MAP = {
    'ace': ('#1f77b4', 'ACE'),
    'dscovr': ('#ff7f0e', 'DSCOVR'),
    'wind': ('#2ca02c', 'WIND'),
}

VAR_LABELS = {
    'Bx': ('Bx', 'nT'),
    'By': ('By', 'nT'),
    'Bz': ('Bz', 'nT'),
    'Ux': ('Vx', 'km/s'),
    'Uy': ('Vy', 'km/s'),
    'Uz': ('Vz', 'km/s'),
    'rho': ('n', 'cm^-3'),
    'T': ('T', 'K'),
}

VARIABLES = list(VAR_LABELS.keys())
COL_HEADERS = [
    'Raw Satellites',
    'Filtered Satellites',
    'Combined',
    'Combined + Propagated',
]

PROP14_STYLE = {'color': '#d62728', 'ls': ':', 'lw': 0.9}
PROP32_STYLE = {'color': '#1f77b4', 'ls': ':', 'lw': 0.9}
COMBINED_STYLE = {'color': '#222', 'ls': '-', 'lw': 0.9}


def fmt_xaxis(ax):
    """Format x-axis as HH:MM for a single-day plot."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))



def _set_shared_ylim(row_axes, series_list, log_scale=False):
    """Set a single y-range across all columns in one row."""
    finite_arrays = []
    for s in series_list:
        if s is None:
            continue
        vals = np.asarray(s, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if log_scale:
            vals = vals[vals > 0]
        if vals.size:
            finite_arrays.append(vals)

    if not finite_arrays:
        return

    all_vals = np.concatenate(finite_arrays)
    y_min = float(all_vals.min())
    y_max = float(all_vals.max())

    if log_scale:
        y0 = y_min * 0.9
        y1 = y_max * 1.1
    else:
        if y_min == y_max:
            pad = 1.0 if y_min == 0 else abs(y_min) * 0.05
        else:
            pad = 0.05 * (y_max - y_min)
        y0 = y_min - pad
        y1 = y_max + pad

    for ax in row_axes:
        ax.set_ylim(y0, y1)


def _read_day_files(day_str):
    """Return raw sats, filtered sats, combined, 14 Re, 32 Re for one day."""
    dt = datetime.strptime(day_str, '%Y-%m-%d')
    filt_dir = dt.strftime('L1/%Y/%m/%d')
    raw_dir = dt.strftime('L1_raw/%Y/%m/%d')

    raw_sats = {}
    filt_sats = {}
    for sat in ('ace', 'dscovr', 'wind'):
        df_raw = read_l1_data(os.path.join(raw_dir, f'L1_{sat}.dat'))
        if not df_raw.empty:
            raw_sats[sat] = df_raw

        df_filt = read_l1_data(os.path.join(filt_dir, f'L1_{sat}.dat'))
        if not df_filt.empty:
            filt_sats[sat] = df_filt

    df_combined = read_l1_data(os.path.join(filt_dir, 'L1_combined.dat'))
    df_14re = read_l1_data(os.path.join(filt_dir, 'IMF_14Re.dat'))
    df_32re = read_l1_data(os.path.join(filt_dir, 'IMF_32Re.dat'))
    return raw_sats, filt_sats, df_combined, df_14re, df_32re


def _plot_row(row_axes, var, raw_sats, filt_sats, df_combined, df_14re, df_32re):
    """Plot one variable row across the four columns."""
    series_for_ylim = []

    # Column 0: raw satellites
    ax = row_axes[0]
    for sat, df in raw_sats.items():
        color, label = COL_MAP[sat]
        if var in df.columns:
            ax.plot(df.index, df[var], lw=0.75, color=color, alpha=0.85, label=label)
            series_for_ylim.append(df[var])

    # Column 1: filtered satellites
    ax = row_axes[1]
    for sat, df in filt_sats.items():
        color, label = COL_MAP[sat]
        if var in df.columns:
            ax.plot(df.index, df[var], lw=0.75, color=color, alpha=0.85, label=label)
            series_for_ylim.append(df[var])

    # Column 2: combined only
    ax = row_axes[2]
    if not df_combined.empty and var in df_combined.columns:
        ax.plot(df_combined.index, df_combined[var], **COMBINED_STYLE)
        series_for_ylim.append(df_combined[var])

    # Column 3: combined + propagated
    ax = row_axes[3]
    if not df_combined.empty and var in df_combined.columns:
        ax.plot(df_combined.index, df_combined[var], **COMBINED_STYLE)
        series_for_ylim.append(df_combined[var])
    if not df_14re.empty and var in df_14re.columns:
        ax.plot(df_14re.index, df_14re[var], **PROP14_STYLE)
        series_for_ylim.append(df_14re[var])
    if not df_32re.empty and var in df_32re.columns:
        ax.plot(df_32re.index, df_32re[var], **PROP32_STYLE)
        series_for_ylim.append(df_32re[var])

    if var == 'T':
        for ax in row_axes:
            ax.set_yscale('log', nonpositive='mask')

    _set_shared_ylim(row_axes, series_for_ylim, log_scale=(var == 'T'))


def _format_row_axes(row_idx, row_axes, n_rows, x_formatter, var):
    for ax in row_axes:
        ax.tick_params(axis='both', labelsize=7)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.grid(True, lw=0.3, alpha=0.5)
        if var != 'T':
            ax.axhline(0, lw=0.5, color='gray', ls='--', alpha=0.5)

    if row_idx < n_rows - 1:
        for ax in row_axes:
            plt.setp(ax.get_xticklabels(), visible=False)
    else:
        for ax in row_axes:
            x_formatter(ax)
            ax.set_xlabel('UT (hr)', fontsize=8)
            plt.setp(ax.get_xticklabels(), fontsize=7, rotation=30, ha='right')


def _add_legends(fig_axes):
    sat_handles = [
        Line2D([0], [0], color=COL_MAP['ace'][0], lw=1.1, label=COL_MAP['ace'][1]),
        Line2D([0], [0], color=COL_MAP['dscovr'][0], lw=1.1, label=COL_MAP['dscovr'][1]),
        Line2D([0], [0], color=COL_MAP['wind'][0], lw=1.1, label=COL_MAP['wind'][1]),
    ]
    fig_axes[0, 0].legend(
        handles=sat_handles,
        fontsize=7,
        loc='upper right',
        framealpha=0.7,
        ncol=3,
        handlelength=1.2,
        columnspacing=0.8,
    )

    prop_handles = [
        Line2D([0], [0], color=COMBINED_STYLE['color'], lw=1.1, ls='-', label='Combined'),
        Line2D([0], [0], color=PROP14_STYLE['color'], lw=1.1, ls=':', label='14 Re'),
        Line2D([0], [0], color=PROP32_STYLE['color'], lw=1.1, ls=':', label='32 Re'),
    ]
    fig_axes[0, 3].legend(
        handles=prop_handles,
        fontsize=7,
        loc='upper right',
        framealpha=0.7,
        ncol=3,
        handlelength=1.2,
        columnspacing=0.8,
    )


# ---------------------------------------------------------------------------
# Per-day plot
# ---------------------------------------------------------------------------

def plot_day(day_str, output_dir='plots'):
    raw_sats, filt_sats, df_combined, df_14re, df_32re = _read_day_files(day_str)

    n_rows = len(VARIABLES)
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(22, 2.2 * n_rows),
        sharex='col',
        gridspec_kw={'hspace': 0.08, 'wspace': 0.28},
    )

    for col_idx, header in enumerate(COL_HEADERS):
        axes[0, col_idx].set_title(header, fontsize=10, fontweight='bold', pad=4)

    for row_idx, var in enumerate(VARIABLES):
        var_label, var_unit = VAR_LABELS[var]
        row_axes = axes[row_idx]
        _plot_row(row_axes, var, raw_sats, filt_sats, df_combined, df_14re, df_32re)
        row_axes[0].set_ylabel(f'{var_label}\n({var_unit})', fontsize=8)
        _format_row_axes(row_idx, row_axes, n_rows, fmt_xaxis, var)

    _add_legends(axes)

    fig.suptitle(f'L1 Solar Wind - {day_str}', fontsize=13, fontweight='bold', y=1.002)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{day_str.replace('-', '_')}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    days = pd.date_range(start='2024-05-01', end='2024-05-31').strftime('%Y-%m-%d').tolist()
    for day in days:
        plot_day(day)
