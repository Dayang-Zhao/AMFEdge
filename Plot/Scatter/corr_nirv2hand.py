import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd

import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns
import GlobVars as gv

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
fig.set_size_inches(cm2inch(7.5), cm2inch(6))

path1 = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
df1 = pd.read_csv(path1).rename(columns={'ID': 'Id'})
path2 = r"F:\Research\AMFEdge\Environment\Amazon_Grids_Hand_stat.csv"
df2 = pd.read_csv(path2)
df = df1.merge(df2, how='left', on='Id')

xcol, ycol = 'HAND_mean', 'nirv_magnitude'

dst_df = df.loc[(~np.isnan(df[xcol]))&(~np.isnan(df[ycol]))]

result = linregress(dst_df[xcol], dst_df[ycol])

axes.scatter(dst_df[xcol], dst_df[ycol], c='#3d98c2', s=20, alpha=0.4)
# axes.plot([0, 1], [0, 1], transform=axes.transAxes, ls='--', c='.3',zorder=-1)
axes.text(0.05, 0.85, '$r$= '+str(result.rvalue.round(2)), fontsize=10, transform=axes.transAxes)
axes.text(0.05, 0.75, '$p$<0.001', fontsize=10, transform=axes.transAxes)
axes.set_xlabel('$\Delta$NIRv Scale (km)', fontsize=LABEL_SIZE)
axes.set_ylabel('$\Delta$NDWI Scale (km)', fontsize=LABEL_SIZE)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.18, right=0.95, hspace=0.55, wspace=0.22)

outpath = r"E:\Thesis\AMFEdge\Figures\Cause\corr_nirv2ndwi_r07.pdf"
fig.savefig(outpath, dpi=600)