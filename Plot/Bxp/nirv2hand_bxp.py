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

path = r"F:\Research\AMFEdge\Model\Amazon_Attribution.csv"
df = pd.read_csv(path)

xcol, ycol = 'HAND_mean', 'nirv_scale'

df[ycol] = df[ycol]/1000
dst_df = df.loc[
    (df['nirv_scale']<=6)&
                (~np.isnan(df[xcol]))&(~np.isnan(df[ycol]))]

# ------------ Group by distance -------------
# bins = [0, 5, 10, 15, 20, 100]
# labels = ['0-5', '5-10', '5-15', '15-20', '>20'] #[str(i) for i in range(len(bins)-1)]
bins = [0, 5, 10, 20, 100]
labels = ['0-5', '5-10', '10-20', '>20']
dst_df['Dist_group'] = pd.cut(dst_df[xcol], bins=bins, labels=labels, right=False)

# ------------- ANOVA ---------------------
# 获取每个 Dist_group 对应的 Scale 值
groups = [dst_df[dst_df['Dist_group'] == label][ycol] for label in labels]

# 执行 ANOVA 方差分析
f_statistic, p_value = stats.f_oneway(*groups)

# ----------------- Plot ------------------
# axes.scatter(dst_df[xcol], dst_df[ycol], c='#3d98c2', s=20, alpha=0.4)
sns.boxplot(x='Dist_group', y=ycol, data=dst_df, hue='Dist_group', palette='flare',
            whis=1, legend=False,
            dodge=False, ax=axes, width=0.5, fliersize=0, linewidth=1.5)

axes.text(0.65, 0.9, '**$F$= '+str(f_statistic.round(2)), fontsize=10+2, transform=axes.transAxes)
# axes.text(0.85, 0.75, '$p$<0.05', fontsize=10, transform=axes.transAxes)
axes.set_ylabel('$\Delta$NIRv Scale (km)', fontsize=LABEL_SIZE)
axes.set_xlabel('HAND (m)', fontsize=LABEL_SIZE)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.18, right=0.95, hspace=0.55, wspace=0.22)

outpath = r"E:\Thesis\AMFEdge\Figures\Cause\bxp_nirv2HAND.pdf"
fig.savefig(outpath, dpi=600)