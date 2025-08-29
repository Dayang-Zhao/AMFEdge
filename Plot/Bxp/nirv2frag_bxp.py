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
import matplotlib.colors as mcolors
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
df['rnirv_scale'] = df['nirv_scale']*1000/df['undistForest_dist']
xcol, ycol = 'undistForest_dist', 'rnirv_scale'

dst_df = df.loc[
    (df['nirv_scale']<=6)&
                (~np.isnan(df[xcol]))&(~np.isnan(df[ycol]))]

# ------------ Group by distance -------------
bins = [0, 3000, 6000, 9000, 16000]
labels = ['0-3', '3-6', '6-9', '>9']
dst_df['Dist_group'] = pd.cut(dst_df[xcol], bins=bins, labels=labels, right=False)

# ------------- ANOVA ---------------------
# 获取每个 Dist_group 对应的 Scale 值
groups = [dst_df[dst_df['Dist_group'] == label][ycol] for label in labels]

# 执行 ANOVA 方差分析
f_statistic, p_value = stats.f_oneway(*groups)

# ----------------- Plot ------------------
palette = {'0-3': '#F3EE72', '3-6': '#C4DA69', '6-9': '#94C466', '>9': '#359464'}
sns.boxplot(x='Dist_group', y=ycol, data=dst_df, hue='Dist_group', palette=palette,
            whis=1, legend=False, dodge=False, linewidth=2, ax=axes, width=0.5, fliersize=0, fill=False, 
            showmeans=True, meanprops=dict(marker='x', markersize=6, markeredgecolor='black'))

# sns.scatterplot(data=dst_df, x='Dist_group', y=ycol, hue='Dist_group', palette=palette, s=20, alpha=0.4,
#                 legend=False, ax=axes)
# axes.text(0.32, 0.9, '$F$= '+str(f_statistic.round(2)), fontsize=12, transform=axes.transAxes)
axes.text(0.32, 0.9, '***$F$= '+str(f_statistic.round(2)), fontsize=12, transform=axes.transAxes)
# axes.text(0.62, 0.9, '$p$='+str(p_value.round(2)), fontsize=12, transform=axes.transAxes)
# axes.set_ylabel('$\Delta$NIRv Magnitude (%)', fontsize=LABEL_SIZE, labelpad=-2)
# axes.set_ylabel('$\Delta$NIRv Scale (km)', fontsize=LABEL_SIZE, labelpad=2)
axes.set_ylabel('$\Delta$NIRv Scale / $d_{95}$ (-)', fontsize=LABEL_SIZE, labelpad=2)
axes.set_xlabel('$d_{95}$ (km)', fontsize=LABEL_SIZE)
axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
axes.axhline(1, 0, 1, color='#ed3e2e', linewidth=2, linestyle='--')
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.18, right=0.95, hspace=0.55, wspace=0.22)

outpath = r"E:\Thesis\AMFEdge\Figures\Cause\rSnirv2d95_bxp.pdf"
fig.savefig(outpath, dpi=600)