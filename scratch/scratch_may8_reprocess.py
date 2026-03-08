"""
scratch_may8_reprocess.py
-------------------------
Re-runs the combine step on the existing May 8 per-satellite .dat files and
plots old vs new combined output side-by-side.  All output goes to scratch/.

Run from the repo root:
    python3 scratch_may8_reprocess.py
"""

import os
import sys
import unittest.mock as mock

# Mock CDF/NetCDF imports so pipeline modules load without those libraries.
for mod in ('netCDF4', 'cdflib'):
    sys.modules.setdefault(mod, mock.MagicMock())

import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
import matplotlib                           # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt             # noqa: E402
import matplotlib.dates as mdates           # noqa: E402

from l1_combine import combine_data_priority  # noqa: E402
from l1_filters import despike               # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAY = '2024-05-08'
DATA_DIR = 'L1/2024/05/08'
SCRATCH  = 'scratch'
os.makedirs(SCRATCH, exist_ok=True)

PLOT_VARS = ['rho', 'T', 'Ux', 'Bz']
VAR_LABELS = {
    'rho': ('n',  'cm⁻³'),
    'T':   ('T',  'K'),
    'Ux':  ('Vx', 'km/s'),
    'Bz':  ('Bz', 'nT'),
}
SAT_COLORS = {'ace': '#1f77b4', 'dscovr': '#ff7f0e', 'wind': '#2ca02c'}

# ---------------------------------------------------------------------------
# Read .dat files
# ---------------------------------------------------------------------------
def read_dat(path):
    col_names = ['year', 'mo', 'dy', 'hr', 'mn', 'sc', 'msc',
                 'Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep=r'\s+', names=col_names,
                     comment='#', skiprows=3, usecols=range(len(col_names)))
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(
        df[['year', 'mo', 'dy', 'hr', 'mn', 'sc']].rename(
            columns={'year':'year','mo':'month','dy':'day',
                     'hr':'hour','mn':'minute','sc':'second'}
        )
    )
    return df.set_index('timestamp')

print('Reading per-satellite data ...')
sats = {s: read_dat(os.path.join(DATA_DIR, f'L1_{s}.dat'))
        for s in ('ace', 'dscovr', 'wind')}
df_old = read_dat(os.path.join(DATA_DIR, 'L1_combined.dat'))

# ---------------------------------------------------------------------------
# Re-run combine
# ---------------------------------------------------------------------------
print('Re-running combine + despike ...')
numeric_cols = ['Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']

master = pd.date_range(start=DAY, periods=1440, freq='1min')

data_map = {}
for sat, df in sats.items():
    if not df.empty:
        data_map[sat] = df[numeric_cols]

df_new, provenance = combine_data_priority(data_map, master)
df_new = despike(df_new)

# ---------------------------------------------------------------------------
# Save new combined to scratch
# ---------------------------------------------------------------------------
out_path = os.path.join(SCRATCH, 'L1_combined_new.dat')
with open(out_path, 'w') as f:
    f.write(f'Re-processed combined L1 for {DAY}\n')
    f.write('year  mo  dy  hr  mn  sc msc Bx By Bz Ux Uy Uz rho T\n')
    f.write('#START\n')
    for t, row in df_new.iterrows():
        if pd.isna(row['Bx']):
            continue
        f.write(
            f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} "
            f"{t.second:2d}   0 "
            f"{row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} "
            f"{row['Ux']:9.2f} {row['Uy']:9.2f} {row['Uz']:9.2f} "
            f"{row['rho']:9.4f} {row['T']:10.1f}\n"
        )
print(f'Saved {out_path}')

# ---------------------------------------------------------------------------
# Plot: old combined vs new combined, with individual sats for context
# ---------------------------------------------------------------------------
print('Plotting ...')
n_rows = len(PLOT_VARS)
fig, axes = plt.subplots(
    n_rows, 2,
    figsize=(14, 2.5 * n_rows),
    sharex=True,
    gridspec_kw={'hspace': 0.08, 'wspace': 0.25},
)

axes[0, 0].set_title('Old Combined', fontsize=10, fontweight='bold')
axes[0, 1].set_title('New Combined (2× threshold)', fontsize=10, fontweight='bold')

for row_idx, var in enumerate(PLOT_VARS):
    label, unit = VAR_LABELS[var]

    for col_idx, df_comb in enumerate([df_old, df_new]):
        ax = axes[row_idx, col_idx]

        # Individual satellites (faint background)
        for sat, df in sats.items():
            if not df.empty and var in df.columns:
                ax.plot(df.index, df[var], lw=0.6, alpha=0.45,
                        color=SAT_COLORS[sat], label=sat.upper() if row_idx == 0 else None)

        # Combined (thick, black)
        if not df_comb.empty and var in df_comb.columns:
            ax.plot(df_comb.index, df_comb[var], lw=1.2,
                    color='black', label='Combined' if row_idx == 0 else None)

        ax.set_ylabel(f'{label}\n({unit})', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        if var in ('Bz', 'Ux', 'Uy', 'Uz'):
            ax.axhline(0, lw=0.5, color='gray', ls='--', alpha=0.6)

    # Share y-limits across old/new for each row
    all_lims = [axes[row_idx, c].get_ylim() for c in range(2)]
    ymin = min(l[0] for l in all_lims)
    ymax = max(l[1] for l in all_lims)
    for c in range(2):
        axes[row_idx, c].set_ylim(ymin, ymax)

# X-axis formatting on the bottom row
for c in range(2):
    ax = axes[-1, c]
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
    ax.set_xlabel(f'{DAY}  UT', fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=7, rotation=30, ha='right')

# Legend on top-left panel
axes[0, 0].legend(fontsize=7, loc='upper right', framealpha=0.7,
                  ncol=4, handlelength=1.2)

fig.suptitle(f'May 8 2024 — Before vs After Quality Threshold Fix',
             fontsize=12, fontweight='bold', y=1.002)

plot_path = os.path.join(SCRATCH, 'may8_before_after.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {plot_path}')
