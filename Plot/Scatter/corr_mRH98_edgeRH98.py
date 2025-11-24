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

def main(dfs:list, grid:tuple, cols:list, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(13), cm2inch(7))

    def _add_r2(ax, linear_result, pos_x, pos_y):
        if linear_result.pvalue < 0.001:
            ax.text(pos_x, pos_y, f"***$r$ = {linear_result.rvalue:.2f}", transform=ax.transAxes, fontsize=LABEL_SIZE+2)
        elif linear_result.pvalue < 0.01:
            ax.text(pos_x, pos_y, f"**$r$ = {linear_result.rvalue:.2f}", transform=ax.transAxes, fontsize=LABEL_SIZE+2)
        elif linear_result.pvalue < 0.05:
            ax.text(pos_x, pos_y, f"*$r$ = {linear_result.rvalue:.2f}", transform=ax.transAxes, fontsize=LABEL_SIZE+2)
        else:
            ax.text(pos_x, pos_y, f"$r$ = {linear_result.rvalue:.2f}", transform=ax.transAxes, fontsize=LABEL_SIZE+2)

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            df = dfs[i*ncols+j]
            xcol, ycol = cols[i*ncols+j]

            # Linear regression.
            result = linregress(df[xcol], df[ycol])

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            cmap = plot_setting['cmaps'][i*ncols+j]
            norm = plot_setting['norms'][i*ncols+j]
            xlabel = plot_setting['xlabels'][i*ncols+j]
            ylabel = plot_setting['ylabels'][i*ncols+j]
            xlim = plot_setting['xlims'][i*ncols+j]
            ylim = plot_setting['ylims'][i*ncols+j]
            extend = plot_setting['extends'][i*ncols+j]

            # Plot.
            df.plot.scatter(ax=ax, x=xcol, y=ycol, color='#299d8f', marker='o', 
                            edgecolor='black', cmap=cmap, alpha=0.8, norm=norm, 
                            legend=True, colorbar=False)
            ## Plot regression line.
            x = np.linspace(xlim[0], xlim[1], 100)
            y = result.intercept + result.slope * x
            ax.plot(x, y, color='black', linewidth=1.5, linestyle='--')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

            # Plugins.
            ax.set_title(title_num+' '+title, loc='left', fontsize=LABEL_SIZE+2)
            if j == 0:
                _add_r2(ax, result, 0.5, 0.8)
            else:
                _add_r2(ax, result, 0.6, 0.3)

            # Remove frames and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=0.18, top=0.9, left=0.12, right=0.98, hspace=0, wspace=0.25)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    path = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_Effect_2023.csv"
    df = pd.read_csv(path)
    df['edge_rh98'] = df['rh98_para3'] + df['rh98_para1']
    df = df.dropna(subset=['rh98_magnitude'])
    df = df[df['rh98_magnitude'] > 0]
    dfs = [df]*2
    cols = [
        ('rh98_magnitude', 'edge_rh98'),
        ('rh98_magnitude', 'rh98_para3')
    ]
    
    # Plot setting.
    cmap2 = mpl.cm.RdBu_r
    levels = np.arange(-8, 9, 2)
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)
    cmaps = [cmap2, cmap2]
    norms = [norm2, norm2]
    xlabels = ['$M_{RH98}$ (m)']*2
    ylabels = ['Edge RH98 (m)', 'Interior RH98 (m)']
    xlims = [(0, 20), (0, 20)]
    ylims = [(10, 35), (18, 35)]
    extend = ['both', 'max']
    titles = ['', '']

    plot_setting = {'cmaps': cmaps,'norms': norms,'extends': extend,'titles': titles,
                    'xlabels': xlabels, 'ylabels': ylabels, 'xlims': xlims, 'ylims': ylims}
    outpath = r"E:\Thesis\AMFEdge\Figures\EdgeRH\corr_Mrh98&EdgeRH98.jpg"
    main(dfs, (1, 2), cols, plot_setting, outpath=outpath)