import pandas as pd
import numpy as np

base = '/Users/connordimarco/Library/CloudStorage/OneDrive-Umich/Work/Propagation/SWMF-IMF/L1/2024/05/01/'

def read_sat(fname, satname):
    cols = ['year','mo','dy','hr','mn','sc','msc','Bx','By','Bz','Ux','Uy','Uz','rho','T']
    with open(base+fname) as fh:
        lines = fh.readlines()
    si = next(i for i,l in enumerate(lines) if l.strip() == '#START')
    df = pd.read_csv(base+fname, sep=r'\s+', skiprows=si+1, names=cols)
    df['time'] = pd.to_datetime(dict(year=df.year, month=df.mo, day=df.dy, hour=df.hr, minute=df.mn))
    df = df.set_index('time')
    print(f"{satname}: {df.shape[0]} rows, NaN per col: { {c:int(df[c].isna().sum()) for c in ['Ux','Uy','Uz','rho','T']} }")
    full_grid = pd.date_range('2024-05-01', periods=1440, freq='1min')
    missing = full_grid.difference(df.index)
    print(f"  Missing minutes from full day: {len(missing)}")
    return df

ace   = read_sat('L1_ace.dat',   'ACE')
dscovr = read_sat('L1_dscovr.dat','DSCOVR')
wind  = read_sat('L1_wind.dat',  'WIND')

# Check whether WIND or ACE have T data at the times when combined T is NaN
comb_cols = ['year','mo','dy','hr','mn','sc','msc','Bx','By','Bz','Ux','Uy','Uz','rho','T','nSat','satUsed']
with open(base+'L1_combined.dat') as fh:
    lines = fh.readlines()
si = next(i for i,l in enumerate(lines) if l.strip() == '#START')
comb = pd.read_csv(base+'L1_combined.dat', sep=r'\s+', skiprows=si+1, names=comb_cols)
comb['time'] = pd.to_datetime(dict(year=comb.year, month=comb.mo, day=comb.dy, hour=comb.hr, minute=comb.mn))
comb = comb.set_index('time')

nan_T = comb[comb['T'].isna()]
print(f"\nCombined NaN T rows: {len(nan_T)}")
print("satUsed breakdown:", nan_T['satUsed'].value_counts().to_dict())

# For those times, what do ACE/WIND/DSCOVR have?
sample_times = nan_T.index[:10]
for t in sample_times:
    ace_T   = ace.loc[t,'T']   if t in ace.index   else 'MISSING'
    dsc_T   = dscovr.loc[t,'T'] if t in dscovr.index else 'MISSING'
    wind_T  = wind.loc[t,'T']  if t in wind.index  else 'MISSING'
    print(f"  {t}  ACE={ace_T}  DSCOVR={dsc_T}  WIND={wind_T}  combined_T={comb.loc[t,'T'] if t in comb.index else '?'}")
