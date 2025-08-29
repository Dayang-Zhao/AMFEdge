import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd

import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
fig.set_size_inches(cm2inch(7.5), cm2inch(7))

path = r"F:\Research\AMFEdge\Model\Amazon_Attribution.csv"
df = pd.read_csv(path)
df['nirv_scale'] = df['nirv_scale']/1000

xcol, ycol, ccol, scol = 'length_mean', 'nirv_magnitude', 'nirv_magnitude', 'nirv_scale'

dst_df = df.loc[(df['nirv_scale']<=6)&(~np.isnan(df[xcol]))&(~np.isnan(df[ycol]))]

result = linregress(dst_df[xcol], dst_df[ycol])

cmap = mpl.cm.RdBu_r
levels = np.arange(-8, 9, 2)
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
dst_df.plot.scatter(ax=axes, x=xcol, y=ycol, c=ccol, s=dst_df[scol] *10, marker='o', 
                edgecolor='black', cmap=cmap, alpha=0.8, norm=norm, 
                legend=True, colorbar=False)

xlim = [0, 8]
x = np.linspace(xlim[0], xlim[1], 100)
y = result.intercept + result.slope * x
axes.plot(x, y, color='black', linewidth=1.5, linestyle='--')
axes.set_xlim(xlim)
axes.text(0.05, 0.08, '***$r$= '+str(result.rvalue.round(2)), fontsize=12, transform=axes.transAxes)
# axes.text(0.05, 0.08, '$p$<0.001', fontsize=12, transform=axes.transAxes)
axes.set_xlabel('DL (month)', fontsize=LABEL_SIZE)
axes.set_ylabel('$\Delta$NIRv Magnitude (%)', fontsize=LABEL_SIZE, labelpad=-2)
axes.set_title('c', loc='left', fontsize=LABEL_SIZE+1)


fig.subplots_adjust(bottom=0.2, top=0.92, left=0.18, right=0.95, hspace=0.55, wspace=0.22)

outpath = r"E:\Thesis\AMFEdge\Figures\Cause\corr_nirv2DL.pdf"
fig.savefig(outpath, dpi=600)