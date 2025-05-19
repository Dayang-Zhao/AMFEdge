import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce

from scipy.stats import zscore
import statsmodels.api as sm

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns
import GlobVars as gv

xcols = ['HAND_mean','SCC_mean', 'anoMCWD_mean', 'vpd_mean', 
         'total_precipitation_sum_mean', 'surface_solar_radiation_downwards_sum_mean',
         'volumetric_soil_water_layer_2_mean'] #['undistForest_dist', 'anoMCWD_mean']
ycol = 'nirv_magnitude'

path1 = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
df1 = pd.read_csv(path1)
path2 = r"F:\Research\AMFEdge\TreeCover\frag_Amazon_2023_dist95.csv"
df2 = pd.read_csv(path2)
path3 = r"F:\Research\AMFEdge\Edge\anoVI_Amazon_UndistEdge_2023.csv"
df3 = pd.read_csv(path3)
df3= df3[['Id', 'MCWD_mean', 'anoMCWD_mean']].groupby('Id').mean().reset_index()
path4 = r"F:\Research\AMFEdge\Environment\Amazon_Grids_Hand_stat.csv"
df4 = pd.read_csv(path4)
path5 = r"F:\Research\AMFEdge\Environment\Amazon_Grids_SoilFern_stat.csv"
df5 = pd.read_csv(path5)
path6 = r"F:\Research\AMFEdge\Meteo\Amazon_Grids_anoMeteo_stat.csv"
df6 = pd.read_csv(path6)
df = reduce(lambda left, right: pd.merge(left, right, on='Id', how='inner'), [df1, df2, df3, df4, df5, df6])
df = df.dropna(subset=ycol)
x_std = df[xcols].apply(zscore)
# dst_x, dst_y = x_std[df['Id'].isin(gv.PGRID_IDS)], df[ycol][df['Id'].isin(gv.PGRID_IDS)]
dst_x, dst_y = x_std, df[ycol]

# Linear regression
X, y = dst_x, dst_y
X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const).fit()
print(model_sm.summary())