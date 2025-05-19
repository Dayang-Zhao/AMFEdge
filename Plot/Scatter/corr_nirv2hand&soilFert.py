import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from functools import reduce
from scipy.stats import linregress

import numpy as np
import pandas as pd

import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import GlobVars as gv

VAR = 'NIRv'
COUNT_COLUMN = VAR+'_count'
MEAN_COLUMN = VAR+'_mean'
MEDIAN_COLUMN = VAR+'_median'
STD_COLUMN = VAR+'_std'

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

fig, axes = plt.subplots(
    nrows=1, ncols=1, 
    )
fig.set_size_inches(cm2inch(8), cm2inch(5))

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

xcol, ycol, ccol = 'HAND_mean','SCC_mean', 'nirv_magnitude'


# axes.scatter(dst_df[xcol], dst_df[ycol], c='#3d98c2', s=20, alpha=0.4)
sc = axes.scatter(df[xcol], df[ycol], c=df[ccol], s=10, cmap='RdBu_r')
axes.set_ylim(-1, 0.5)
axes.set_xlim(0, 30)
axes.set_ylabel('log$_{10}$(SoilFert) (cmol/kg)', fontsize=10)
axes.set_xlabel('Hand (m)', fontsize=LABEL_SIZE)
# Set ytick label.
yticks = np.arange(-1, 0.6, 0.2)
axes.set_yticks(yticks)
# axes.set_yticklabels([fr'$10^{{{ytick:.1f}}}$' for ytick in yticks], fontsize=LABEL_SIZE)
# Set cbar label
cbar = plt.colorbar(sc, ax=axes)
cbar.set_label('$\Delta$NIRv Magnitude (%)', fontsize=10)


fig.subplots_adjust(bottom=0.2, top=0.95, left=0.2, right=0.9, hspace=0.55, wspace=0.22)

outpath = r"E:\Thesis\AMFEdge\Figures\Cause\corr_nirv_mg2hand&soil.pdf"
fig.savefig(outpath, dpi=600)