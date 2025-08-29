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
fig.set_size_inches(cm2inch(7), cm2inch(6))

path = r"F:\Research\AMFEdge\Model\Amazon_Attribution.csv"
df = pd.read_csv(path)

df['rnirv_scale'] = df['nirv_scale']/df['undistForest_dist']
df['nirv_scale'] = df['nirv_scale']/1000
df['undistForest_dist'] = df['undistForest_dist']/1000
dst_df = df.loc[(df['nirv_scale']>6)]
dst_df['type'] = 'Outlier'

# ----------------- Plot ------------------
titles = ['(a)', '(b)']
ycols = ['undistForest_dist', 'rnirv_scale']
ylabels = ['$d_{95}$ (km)', '$\Delta$NIRv Scale / $d_{95}$ (-)']
ylims = [(0,7), (0,4)]
for i in range(1):
    
    ax = axes
    ycol = ycols[i]
    ylabel = ylabels[i]
    title = titles[i]
    ylim = ylims[i]

    sns.boxplot(dst_df, x='type', y=ycol, color='#3d98c2', ax=axes, width=0.5, fill=False,
                fliersize=0, linewidth=2, dodge=False, whis=1,
                showmeans=True, meanprops=dict(marker='x', markeredgecolor='black', markersize=8))
    # sns.boxplot(x='type', y=ycol, data=dst_df, color='#3d98c2',
    #             whis=1, legend=False,
    #             dodge=False, ax=ax, width=0.5, fliersize=0, linewidth=1.5)

    if i == 1:
        ax.axhline(1, 0, 1, color='#ed3e2e', linewidth=2, linestyle='--')
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_xlabel(None, fontsize=LABEL_SIZE)
    ax.set_xticks([])
    ax.set_ylim(ylim)
    # ax.set_title(title, fontsize=LABEL_SIZE, loc='left')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

fig.tight_layout()

outpath = r"E:\Thesis\AMFEdge\Figures\Cause\outlier_rSnirv2d95_bxp.jpg"
fig.savefig(outpath, dpi=600)