"""
plot_l1_may2024.py
------------------
For each day in May 2024, reads the L1 .dat files and produces a multi-panel
figure with:
  Rows    : each physical variable (Bx, By, Bz, Ux, Uy, Uz, rho, T)
  Columns : (1) raw satellite data  |  (2) combined  |  (3) 14 Re  |  (4) 32 Re

All three satellites (ACE, DSCOVR, WIND) are overlaid in the first column.
Output is saved to plots/YYYY_MM_DD.png.
"""

from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COL_MAP = {
    'ace':    ('#1f77b4', 'ACE'),
    'dscovr': ('#ff7f0e', 'DSCOVR'),
    'wind':   ('#2ca02c', 'WIND'),
}

VAR_LABELS = {
    'Bx': ('Bx',  'nT'),
    'By': ('By',  'nT'),
    'Bz': ('Bz',  'nT'),
    'Ux': ('Vx',  'km/s'),
    'Uy': ('Vy',  'km/s'),
    'Uz': ('Vz',  'km/s'),
    'rho': ('n',   'cm\u207b\u00b3'),
    'T':  ('T',   'K'),
}

VARIABLES = list(VAR_LABELS.keys())
COL_HEADERS = ['Individual Satellites', 'Combined', '14 R\u2091', '32 R\u2091']


def read_dat(filepath):
    """Read one of the .dat files produced by l1_routines."""
    col_names = ['year', 'mo', 'dy', 'hr', 'mn', 'sc', 'msc',
                 'Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, sep=r'\s+', names=col_names,
                         comment='#', skiprows=3, usecols=range(len(col_names)))
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    dt_cols = df[['year', 'mo', 'dy', 'hr', 'mn', 'sc']].copy()
    dt_cols.columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
    df['timestamp'] = pd.to_datetime(
        dt_cols) + pd.to_timedelta(df['msc'], unit='ms')
    return df.set_index('timestamp')


def fmt_xaxis(ax, day_str):
    """Format x-axis as HH:MM for a single-day plot."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))


def fmt_xaxis_context(ax):
    """Format x-axis as 'DD HH:MM' for a 36-hour context plot."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))


# ---------------------------------------------------------------------------
# Per-day plot
# ---------------------------------------------------------------------------

def plot_day(day_str, output_dir='plots'):
    dt = datetime.strptime(day_str, '%Y-%m-%d')
    data_dir = dt.strftime('L1/%Y/%m/%d')

    # Read all satellite files
    sats = {}
    for sat in ('ace', 'dscovr', 'wind'):
        fpath = os.path.join(data_dir, f'L1_{sat}.dat')
        df = read_dat(fpath)
        if not df.empty:
            sats[sat] = df

    df_combined = read_dat(os.path.join(data_dir, 'L1_combined.dat'))
    df_14re = read_dat(os.path.join(data_dir, 'IMF_14Re.dat'))
    df_32re = read_dat(os.path.join(data_dir, 'IMF_32Re.dat'))

    n_rows = len(VARIABLES)
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(22, 2.2 * n_rows),
        sharex='col',
        gridspec_kw={'hspace': 0.08, 'wspace': 0.28},
    )

    # Column header labels (top row only)
    for col_idx, header in enumerate(COL_HEADERS):
        axes[0, col_idx].set_title(
            header, fontsize=10, fontweight='bold', pad=4)

    for row_idx, var in enumerate(VARIABLES):
        var_label, var_unit = VAR_LABELS[var]
        row_axes = axes[row_idx]

        # ---- Column 0: individual satellites --------------------------------
        ax = row_axes[0]
        for sat, df in sats.items():
            color, label = COL_MAP[sat]
            if var in df.columns:
                ax.plot(df.index, df[var], lw=0.7, color=color,
                        label=label, alpha=0.85)
        ax.set_ylabel(f'{var_label}\n({var_unit})', fontsize=8)

        # ---- Column 1: combined --------------------------------------------
        ax = row_axes[1]
        if not df_combined.empty and var in df_combined.columns:
            ax.plot(df_combined.index, df_combined[var],
                    lw=0.8, color='#444', label='Combined')

        # ---- Column 2: 14 Re -----------------------------------------------
        ax = row_axes[2]
        if not df_14re.empty and var in df_14re.columns:
            ax.plot(df_14re.index, df_14re[var],
                    lw=0.8, color='#9467bd', label='14 R\u2091')

        # ---- Column 3: 32 Re -----------------------------------------------
        ax = row_axes[3]
        if not df_32re.empty and var in df_32re.columns:
            ax.plot(df_32re.index, df_32re[var],
                    lw=0.8, color='#d62728', label='32 R\u2091')

        # Share y-limits: use column 0's auto range for all columns
        ylim = row_axes[0].get_ylim()
        for ax in row_axes[1:]:
            ax.set_ylim(ylim)

        # Common axis formatting
        for ax in row_axes:
            ax.tick_params(axis='both', labelsize=7)
            ax.yaxis.set_tick_params(labelleft=True)
            # Thin grid
            ax.grid(True, lw=0.3, alpha=0.5)
            ax.axhline(0, lw=0.5, color='gray', ls='--', alpha=0.5)

        # Only show x-tick labels on the bottom row
        if row_idx < n_rows - 1:
            for ax in row_axes:
                plt.setp(ax.get_xticklabels(), visible=False)
        else:
            for ax in row_axes:
                fmt_xaxis(ax, day_str)
                ax.set_xlabel('UT (hr)', fontsize=8)
                plt.setp(ax.get_xticklabels(), fontsize=7,
                         rotation=30, ha='right')

    # Satellite legend on column 0, top row
    axes[0, 0].legend(fontsize=7, loc='upper right', framealpha=0.7,
                      ncol=3, handlelength=1.2, columnspacing=0.8)

    # Overall title
    fig.suptitle(f'L1 Solar Wind  \u2013  {day_str}',
                 fontsize=13, fontweight='bold', y=1.002)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{day_str.replace('-', '_')}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Per-day context plot  (±6 h of neighbouring days)
# ---------------------------------------------------------------------------

def _read_day_files(day_str):
    """Return (sats_dict, df_combined, df_14re, df_32re) for one calendar day."""
    dt = datetime.strptime(day_str, '%Y-%m-%d')
    data_dir = dt.strftime('L1/%Y/%m/%d')
    sats = {}
    for sat in ('ace', 'dscovr', 'wind'):
        df = read_dat(os.path.join(data_dir, f'L1_{sat}.dat'))
        if not df.empty:
            sats[sat] = df
    df_combined = read_dat(os.path.join(data_dir, 'L1_combined.dat'))
    df_14re = read_dat(os.path.join(data_dir, 'IMF_14Re.dat'))
    df_32re = read_dat(os.path.join(data_dir, 'IMF_32Re.dat'))
    return sats, df_combined, df_14re, df_32re


def plot_day_with_context(day_str, context_hours=6, output_dir='plots/with_context'):
    """
    Plot ``day_str`` with ``context_hours`` of the previous and following day
    included.  Vertical dashed lines mark the day boundaries so day-to-day
    continuity (or jumps) are immediately visible.
    """
    dt = pd.Timestamp(day_str)
    prev_str = (dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    next_str = (dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    t_start = dt - pd.Timedelta(hours=context_hours)
    t_end = dt + pd.Timedelta(hours=24 + context_hours)

    # Load the three days and concatenate each data stream
    def _concat_sats(*day_strs):
        combined = {}
        for ds in day_strs:
            s, _, _, _ = _read_day_files(ds)
            for sat, df in s.items():
                combined.setdefault(sat, []).append(df)
        return {sat: pd.concat(frames).sort_index() for sat, frames in combined.items()}

    def _concat_df(getter_idx):
        frames = []
        for ds in (prev_str, day_str, next_str):
            parts = _read_day_files(ds)
            df = parts[getter_idx]          # index 1=combined, 2=14re, 3=32re
            if not df.empty:
                frames.append(df)
        return pd.concat(frames).sort_index() if frames else pd.DataFrame()

    sats = _concat_sats(prev_str, day_str, next_str)
    df_combined = _concat_df(1)
    df_14re = _concat_df(2)
    df_32re = _concat_df(3)

    # Clip every stream to the window
    def _clip(df):
        if df.empty:
            return df
        return df.loc[(df.index >= t_start) & (df.index <= t_end)]

    sats = {s: _clip(df) for s, df in sats.items() if not _clip(df).empty}
    df_combined = _clip(df_combined)
    df_14re = _clip(df_14re)
    df_32re = _clip(df_32re)

    n_rows = len(VARIABLES)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(22, 2.2 * n_rows),
        sharex='col',
        gridspec_kw={'hspace': 0.08, 'wspace': 0.28},
    )

    # Column header labels
    for col_idx, header in enumerate(COL_HEADERS):
        axes[0, col_idx].set_title(
            header, fontsize=10, fontweight='bold', pad=4)

    # Boundary timestamps for vertical lines
    boundaries = [dt, dt + pd.Timedelta(days=1)]

    for row_idx, var in enumerate(VARIABLES):
        var_label, var_unit = VAR_LABELS[var]
        row_axes = axes[row_idx]

        # ---- Column 0: individual satellites --------------------------------
        ax = row_axes[0]
        for sat, df in sats.items():
            color, label = COL_MAP[sat]
            if var in df.columns:
                ax.plot(df.index, df[var], lw=0.7, color=color,
                        label=label, alpha=0.85)
        ax.set_ylabel(f'{var_label}\n({var_unit})', fontsize=8)

        # ---- Column 1: combined --------------------------------------------
        ax = row_axes[1]
        if not df_combined.empty and var in df_combined.columns:
            ax.plot(df_combined.index, df_combined[var],
                    lw=0.8, color='#444', label='Combined')

        # ---- Column 2: 14 Re -----------------------------------------------
        ax = row_axes[2]
        if not df_14re.empty and var in df_14re.columns:
            ax.plot(df_14re.index, df_14re[var],
                    lw=0.8, color='#9467bd', label='14 R\u2091')

        # ---- Column 3: 32 Re -----------------------------------------------
        ax = row_axes[3]
        if not df_32re.empty and var in df_32re.columns:
            ax.plot(df_32re.index, df_32re[var],
                    lw=0.8, color='#d62728', label='32 R\u2091')

        # Share y-limits: use column 0's auto range for all columns
        ylim = row_axes[0].get_ylim()
        for ax in row_axes[1:]:
            ax.set_ylim(ylim)

        for ax in row_axes:
            # Day-boundary vertical lines
            for boundary in boundaries:
                ax.axvline(boundary, color='k', lw=1.0, ls='--', alpha=0.6,
                           zorder=5)
            # Shading for the context regions
            ax.axvspan(t_start, dt, alpha=0.06, color='steelblue', zorder=0)
            ax.axvspan(dt + pd.Timedelta(days=1), t_end,
                       alpha=0.06, color='steelblue', zorder=0)
            ax.tick_params(axis='both', labelsize=7)
            ax.yaxis.set_tick_params(labelleft=True)
            ax.grid(True, lw=0.3, alpha=0.5)
            ax.axhline(0, lw=0.5, color='gray', ls='--', alpha=0.5)
            ax.set_xlim(t_start, t_end)

        if row_idx < n_rows - 1:
            for ax in row_axes:
                plt.setp(ax.get_xticklabels(), visible=False)
        else:
            for ax in row_axes:
                fmt_xaxis_context(ax)
                ax.set_xlabel('Day HH:MM (UT)', fontsize=8)
                plt.setp(ax.get_xticklabels(), fontsize=7,
                         rotation=35, ha='right')

    axes[0, 0].legend(fontsize=7, loc='upper right', framealpha=0.7,
                      ncol=3, handlelength=1.2, columnspacing=0.8)

    fig.suptitle(
        f'L1 Solar Wind \u2013 {day_str}  [\u00b1{context_hours} h context]',
        fontsize=13, fontweight='bold', y=1.002,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{day_str.replace('-', '_')}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    days = pd.date_range(start='2024-05-01',
                         end='2024-05-31').strftime('%Y-%m-%d').tolist()
    for day in days:
        try:
            plot_day(day)
        except Exception as exc:
            print(f"  ERROR for {day}: {exc}")
