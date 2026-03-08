from l1_quality import score_all_plasma
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')


# Parse .dat files directly — no netCDF4 needed

def read_dat(path):
    rows = []
    started = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('#START'):
                started = True
                continue
            if not started or s.startswith('year') or s.startswith('Combined') or s.startswith('Produced'):
                continue
            parts = s.split()
            if len(parts) < 15:
                continue
            yr, mo, dy, hr, mn = int(parts[0]), int(parts[1]), int(
                parts[2]), int(parts[3]), int(parts[4])
            t = pd.Timestamp(yr, mo, dy, hr, mn)
            vals = [float(x) for x in parts[7:15]]
            rows.append([t] + vals)
    cols = ['time', 'Bx', 'By', 'Bz', 'Ux', 'Uy', 'Uz', 'rho', 'T']
    df = pd.DataFrame(rows, columns=cols).set_index('time')
    return df


dsc = read_dat('L1/2024/05/08/L1_dscovr.dat')
ace = read_dat('L1/2024/05/08/L1_ace.dat')
wind = read_dat('L1/2024/05/08/L1_wind.dat')
comb = read_dat('L1/2024/05/08/L1_combined.dat')

print('=== May 8 rho stats ===')
for name, df in [('DSCOVR', dsc), ('ACE', ace), ('WIND', wind), ('COMBINED', comb)]:
    print(f'  {name:8s}: mean={df.rho.mean():.2f}, max={df.rho.max():.2f}, '
          f'rows>10: {(df.rho > 10).sum()}')

print()
print('=== May 8 T stats ===')
for name, df in [('DSCOVR', dsc), ('ACE', ace), ('WIND', wind), ('COMBINED', comb)]:
    print(f'  {name:8s}: mean={df["T"].mean():.0f}, max={df["T"].max():.0f}, '
          f'rows>400k: {(df["T"] > 4e5).sum()}')

# Identify spike times in the combined output
spikes = comb[(comb.rho > 8) | (comb['T'] > 4e5)].copy()
print(f'\nCombined spike minutes (rho>8 or T>400k): {len(spikes)}')

# Build a common master grid (ACE index) and align all satellites to it
master = ace.index
dsc_r = dsc.reindex(master)
wind_r = wind.reindex(master)

print('\n=== Running quality scoring ===')
all_bad = score_all_plasma(ace, dsc_r, wind_r)

for code, name in [(1, 'ACE'), (2, 'DSCOVR'), (3, 'WIND')]:
    for var in ['rho', 'T']:
        n = int(all_bad[code].get(var, pd.Series(False, index=master)).sum())
        print(f'  {name} {var}: {n} minutes flagged bad')

# At each spike: show per-sat values, quality flags, and n_sat
# to determine whether the median path or the 2-sat path is responsible
print('\n=== Per-minute trace at combined spike times ===')
print(f'{"Time":<20} {"nSat":>4} {"DSC_rho":>8} {"ACE_rho":>8} {"WIN_rho":>8} '
      f'{"med_rho":>8} {"C_rho":>8} '
      f'{"DSC_T":>9} {"ACE_T":>9} {"WIN_T":>9} {"C_T":>9} '
      f'{"rho_bad(D/A/W)":>14} {"T_bad(D/A/W)":>13}')

for t in spikes.index:
    if t not in master:
        continue
    d_rho = dsc_r['rho'].get(t, np.nan)
    a_rho = ace['rho'].get(t, np.nan)
    w_rho = wind_r['rho'].get(t, np.nan)
    c_rho = comb['rho'].get(t, np.nan)
    d_T = dsc_r['T'].get(t, np.nan)
    a_T = ace['T'].get(t, np.nan)
    w_T = wind_r['T'].get(t, np.nan)
    c_T = comb['T'].get(t, np.nan)

    d_bad = bool(all_bad[2]['rho'].get(t, False))
    a_bad = bool(all_bad[1]['rho'].get(t, False))
    w_bad = bool(all_bad[3]['rho'].get(t, False))
    d_T_bad = bool(all_bad[2]['T'].get(t, False))
    a_T_bad = bool(all_bad[1]['T'].get(t, False))
    w_T_bad = bool(all_bad[3]['T'].get(t, False))

    # Count valid (non-NaN, not bad) satellites
    valid = []
    if pd.notna(d_rho) and not d_bad:
        valid.append(d_rho)
    if pd.notna(a_rho) and not a_bad:
        valid.append(a_rho)
    if pd.notna(w_rho) and not w_bad:
        valid.append(w_rho)

    med = np.median(valid) if valid else np.nan

    T_valid = []
    if pd.notna(d_T) and not d_T_bad:
        T_valid.append(d_T)
    if pd.notna(a_T) and not a_T_bad:
        T_valid.append(a_T)
    if pd.notna(w_T) and not w_T_bad:
        T_valid.append(w_T)
    med_T = np.median(T_valid) if T_valid else np.nan

    rho_flags = f'{d_bad}/{a_bad}/{w_bad}'
    T_flags = f'{d_T_bad}/{a_T_bad}/{w_T_bad}'
    print(f'{str(t):<20} {len(valid):>4} {d_rho:>8.2f} {a_rho:>8.2f} {w_rho:>8.2f} '
          f'{med:>8.2f} {c_rho:>8.2f} '
          f'{d_T:>9.0f} {a_T:>9.0f} {w_T:>9.0f} {c_T:>9.0f} '
          f'{rho_flags:>14} {T_flags:>13}')
