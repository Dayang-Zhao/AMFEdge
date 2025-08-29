import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rootdir = r"F:\Research\AMFEdge\CMIP6\GEE\SSP2_45"

years = range(2020, 2100)
dfs = []
for year in years:
    fname = os.path.join(rootdir, f"anoMeteo_CMIP6_{year}.csv")
    df = pd.read_csv(fname)
    df['year'] = year
    dfs.append(df)

outdf = pd.concat(dfs, ignore_index=True)
outdf.replace(-9999, np.nan, inplace=True)
outdf2 = outdf[['Id', 'MCWD_mean', 'TreeCover', 'WD_mean',  'surface_solar_radiation_downwards_sum_mean', 
                'temperature_2m_mean', 'total_evaporation_sum_mean', 
                'total_precipitation_sum_mean', 'vpd_mean', 'year']].groupby(['year']).mean().reset_index()

print(outdf)