import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce

from scipy.stats import zscore
import scipy.sparse

def to_array(self):
    return self.toarray()

scipy.sparse.spmatrix.A = property(to_array)
from pygam import LinearGAM, s, f, te

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns
import GlobVars as gv

xcols = ['HAND_mean','SCC_mean', 'undistForest_dist', 
         'surface_solar_radiation_downwards_sum_mean','vpd_mean', 
         'total_precipitation_sum_mean'] #['undistForest_dist', 'anoMCWD_mean']
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
df = df.dropna(subset=xcols+[ycol])
x_std = df[xcols].apply(zscore)
# dst_x, dst_y = x_std[df['Id'].isin(gv.PGRID_IDS)], df[ycol][df['Id'].isin(gv.PGRID_IDS)]
X, y = x_std.values, df[ycol].values

# Fit a GAM model with smooth splines on each variable
gam = LinearGAM(s(0) + s(1) + s(2) + s(3)+s(4) + s(5) + s(6)
                + te(0,1)).fit(X, y)

# Predict
y_pred = gam.predict(X)

# Plot partial dependencies
rows, cols = 2, 4
fig, axs = plt.subplots(rows, cols)
for i in range(rows):
    for j in range(cols):
        ax = axs[i, j]
        term_i = i * cols + j
        XX = gam.generate_X_grid(term=term_i)
        ax.plot(XX[:, term_i], gam.partial_dependence(term=term_i, X=XX))
        ax.plot(XX[:, term_i], gam.partial_dependence(term=term_i, X=XX, width=0.95)[1], c='r', ls='--')
        ax.set_title(f'Effect of {xcols[term_i]}')
plt.tight_layout()
plt.show()