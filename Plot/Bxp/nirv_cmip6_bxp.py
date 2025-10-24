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

nrows, ncols = 1, 2
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, 
    )
fig.set_size_inches(cm2inch(15), cm2inch(6.5))

scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']
scenarios2 = ['RCP 2.6', 'RCP 4.5', 'RCP 8.5']
dfs = []
for scenario in scenarios:
    path = rf"F:\Research\AMFEdge\CMIP6\Predict\QDM\Mnirv_Edge_pred_{scenario}.csv"
    df = pd.read_csv(path)
    df['scenario'] = scenario
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df["scenario"] = df["scenario"].replace({
    "SSP1_26": "RCP 2.6",
    "SSP2_45": "RCP 4.5",
    "SSP5_85": "RCP 8.5"
})
pos_df, neg_df = df[df['nirv_magnitude']>0], df[df['nirv_magnitude']<=0]
data = [pos_df, neg_df]
xcol, ycol = 'scenario', 'nirv_magnitude'
ylims = [(0,6), (-5,-1)]
titles = [r'a $M_{\Delta \mathrm{NIRv}}$ > 0', r'b $M_{\Delta \mathrm{NIRv}}$ < 0']

for j in range(ncols):
    ax = axes[j]
    dst_data = data[j]
    ylim = ylims[j]
    # ------------- ANOVA ---------------------
    # 获取每个 Dist_group 对应的 Scale 值
    groups = [dst_data[dst_data['scenario'] == scenario][ycol] for scenario in scenarios2]

    # 执行 ANOVA 方差分析
    f_statistic, p_value = stats.f_oneway(*groups)
    ax.text(0.32, 0.9, '***$F$= '+str(f_statistic.round(2)), fontsize=12, transform=ax.transAxes)
    # ----------------- Plot ------------------
    palette = dict(zip(scenarios2, ['#576fa0', '#e3b87f','#b57979']))
    # sns.boxplot(x='scenario', y=ycol, data=dst_data, hue='scenario', palette=palette,
    #             whis=1, legend=False, dodge=False, linewidth=2, ax=ax, width=0.5, fliersize=0, fill=False, 
    #             showmeans=True, meanprops=dict(marker='x', markersize=6, markeredgecolor='black'))
    sns.pointplot(data=dst_data, x='scenario', y=ycol, hue='scenario', palette=palette, 
                  errorbar='sd', alpha=1,legend=False, ax=ax)
    # Remove frames and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel('$M_{\Delta \mathrm{NIRv}}$ (%)', fontsize=LABEL_SIZE, labelpad=2)
    ax.set_ylim(ylim)
    ax.set_xlabel(None, fontsize=LABEL_SIZE)
    ax.set_title(titles[j], loc='left', fontsize=LABEL_SIZE+1)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    fig.subplots_adjust(bottom=0.13, top=0.9, left=0.1, right=0.98, hspace=0.55, wspace=0.22)

outpath = r"E:\Thesis\AMFEdge\Figures\CMIP6\nirv_cmip6_bxp.jpg"
fig.savefig(outpath, dpi=600)