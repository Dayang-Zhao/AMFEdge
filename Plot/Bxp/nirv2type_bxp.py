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

types = ['grass', 'water', 'crop', 'nonveg']
labels = ['Grass', 'Water', 'Crop', 'Barren']

def cm2inch(value):
    return value/2.54

def main(dfs:list, grid:tuple, cols:list, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(15), cm2inch(6))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            df = dfs[i*ncols+j]
            xcol, ycol = cols[i*ncols+j]
            ylim = plot_setting['ylims'][i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            text_pos = plot_setting['text_pos'][i*ncols+j]

            def _add_rp(anova_results):
                r, p = anova_results
                if p<0.001:
                    return f'***$F$ = {r.round(3)}'
                elif p<0.01:
                    return f'**$F$ = {r.round(3)}'
                elif p<0.05:
                    return f'*$F$ = {r.round(3)}'
                else:
                    return f'$F$ = {r.round(3)}'
            # ------------- ANOVA ---------------------
            groups = [df[df['type'] == t][ycol] for t in labels]

            # 执行 ANOVA 方差分析
            anova_result = stats.f_oneway(*groups)

            # ----------------- Plot ------------------
            group_colors = dict(zip(labels,
                                 ['#299d8f', '#3076CC', '#f4b41a', "#CC4348",]))
            sns.boxplot(x='type', y=ycol, data=df, hue='type', palette=group_colors,
                        whis=1, legend=False, dodge=False, ax=ax, width=0.5, fliersize=0, 
                        linewidth=2, fill=False, 
                        showmeans=True, meanprops=dict(marker='x', markersize=6, markeredgecolor='black'))

            ax.text(text_pos[0], text_pos[1], _add_rp(anova_result), fontsize=10+2, transform=ax.transAxes)
            ax.set_ylabel('$M_{\Delta \mathrm{NIRv}}$ (%)', fontsize=LABEL_SIZE)
            ax.set_xlabel('', fontsize=LABEL_SIZE)
            ax.set_ylim(ylim)
            ax.set_title(title_nums[i*ncols+j]+' '+title, loc='left', fontsize=LABEL_SIZE+2)

            # Remove frames and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.08, right=0.98, hspace=0.55, wspace=0.22)
    fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    ycol = 'dNIRv_mean'
    path = r"F:\Research\AMFEdge\EdgeType\anoVI_Amazon_specUndistEdge_effect_2023.csv"
    df = pd.read_csv(path)
    df[ycol] = df[ycol]*-1
    df['type'] = df['type'].map(dict(zip(types, labels)))
    df = df[df['dNIRv_mean'].notna()]
    pos_df = df[df[ycol]>=0]
    neg_df = df[df[ycol]<0]
    dfs = [pos_df, neg_df]

    grid = (1, 2)
    cols = [('type', ycol)]*2
    ylim = [(0, 8), (-6, 0)]
    titles= ['$M_{\Delta \mathrm{NIRv}}$ > 0', '$M_{\Delta \mathrm{NIRv}}$ < 0']
    text_pos = [(0.65, 0.92), (0.65, 0.08)]
    plot_setting = {'ylims': ylim, 'titles': titles, 'text_pos': text_pos}

    
    outpath = r"E:\Thesis\AMFEdge\Figures\EdgeType\bxp_nirv2type.jpg"
    main(dfs=dfs, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
    

