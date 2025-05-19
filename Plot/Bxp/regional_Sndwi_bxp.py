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
import matplotlib.colors as mcolors
import seaborn as sns
import GlobVars as gv

y = 'ndwi_scale'

LABEL_SIZE = 16
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

path = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
df = pd.read_csv(path)
df = df.loc[df[y].notna(),:]
df['class'] = df['Id'].isin(gv.NEGRID_IDS).map({True: 1, False: 2})
df['ndwi_scale'] = df['ndwi_scale']/1000

# Mann–Whitney U Test.
data1 = df.loc[df['class']==1, y].dropna()
data2 = df.loc[df['class']==2, y].dropna()
stat, p = stats.mannwhitneyu(data1, data2)

palette = {'1': '#3d98c2','2': '#ed3e2e'}
sns.boxplot(df, x='class', y=y, palette=palette, ax=axes, width=0.4,)

for patch, cat in zip(axes.patches, sorted(df['class'].unique())):
    color = palette[str(cat)]
    patch.set_edgecolor(color)     # 设置边框颜色
    patch.set_facecolor(mcolors.to_rgba(color, alpha=0.6))
    # patch.set_alpha(0.6)           # 设置填充透明度
    # patch.set_facecolor(color)     # 设置填充颜色

    patch.set_linewidth(2)  

for i, cat in enumerate(sorted(df['class'].unique())):
    color = palette[str(cat)]
    lines = axes.lines[i*6 : i*6 + 5]  # 取 whiskers(2), caps(2), median(1)
    
    # whiskers
    lines[0].set_color(color)
    lines[0].set_linewidth(2)
    lines[1].set_color(color)
    lines[1].set_linewidth(2)
    
    # caps
    lines[2].set_color(color)
    lines[2].set_linewidth(2)
    lines[3].set_color(color)
    lines[3].set_linewidth(2)
    
    # median
    lines[4].set_color(color)
    lines[4].set_linewidth(2)

# axes.set_ylabel('$\Delta$NIRv Scale (km)', fontsize=LABEL_SIZE)
axes.set_ylabel(None)
axes.set_xlabel(None)
axes.set_xticklabels(['NE', 'SW'], fontsize=LABEL_SIZE, rotation=0)

# Add text
axes.text(0.05, 0.5, f'n={len(data1)}', fontsize=LABEL_SIZE, transform=axes.transAxes)
axes.text(0.7, 0.15, f'n={len(data2)}', fontsize=LABEL_SIZE, transform=axes.transAxes)
axes.text(0.3, 0.75, f'**$U$={stat}', fontsize=LABEL_SIZE, transform=axes.transAxes)

fig.subplots_adjust(bottom=0.15, top=0.95, left=0.12, right=0.95, hspace=0.55, wspace=0.22)

outpath = r"E:\Thesis\AMFEdge\Figures\Edge\ndwi_scale_bxp.pdf"
fig.savefig(outpath, dpi=600)