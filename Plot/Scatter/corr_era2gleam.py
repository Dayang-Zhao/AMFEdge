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
    title_nums = ['c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(13), cm2inch(7))

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
            ax.text(0.05, 0.90, f"***$r$ = {result.rvalue:.2f}", transform=ax.transAxes, fontsize=LABEL_SIZE+2)
            # ax.text(0.05, 0.80, f"$p$ < 0.001", transform=ax.transAxes, fontsize=LABEL_SIZE+2)
            # ax.text(0.05, 0.70, f"slope = {result.slope:.2f}", transform=ax.transAxes, fontsize=LABEL_SIZE+2)

            # Remove frames and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=0.18, top=0.9, left=0.12, right=0.98, hspace=0, wspace=0.25)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    path1 = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_Edge_Effect_2023.csv"
    df1 = pd.read_csv(path1)
    path2 = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_GLEAM_Edge_Effect_2023.csv"
    df2 = pd.read_csv(path2)
    df = pd.merge(df1, df2, on=['Id'], suffixes=('_era5', '_gleam'))
    df[['nirv_scale_era5', 'nirv_scale_gleam']] = df[['nirv_scale_era5', 'nirv_scale_gleam']]/1000
    dst_df = df.loc[
        (df['nirv_scale_era5']!=6.001)&(df['nirv_scale_gleam']!=6.001)&
                    (~np.isnan(df['nirv_scale_era5']))&(~np.isnan(df['nirv_scale_gleam']))]
    dfs = [dst_df]*2
    cols = [
        ('nirv_magnitude_gleam', 'nirv_magnitude_era5'),
        ('nirv_scale_gleam', 'nirv_scale_era5')
    ]
    
    # Plot setting.
    cmap2 = mpl.cm.RdBu_r
    levels = np.arange(-8, 9, 2)
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)
    cmaps = [cmap2, cmap2]
    norms = [norm2, norm2]
    xlabels = [r'GLEAM $M_{\Delta \mathrm{NIRv}}$ (%)', r'GLEAM $S_{\Delta \mathrm{NIRv}}$ (km)']
    ylabels = [r'ERA5 $M_{\Delta \mathrm{NIRv}}$ (%)', r'ERA5 $S_{\Delta \mathrm{NIRv}}$ (km)']
    xlims = [(-8, 10), (0, 6.5)]
    ylims = [(-8, 10), (0, 6.5)]
    extend = ['both', 'max']
    titles = ['', '']

    plot_setting = {'cmaps': cmaps,'norms': norms,'extends': extend,'titles': titles,
                    'xlabels': xlabels, 'ylabels': ylabels, 'xlims': xlims, 'ylims': ylims}
    outpath = r"E:\Thesis\AMFEdge\Figures\Comparison\corr_era2gleam.pdf"
    main(dfs, (1, 2), cols, plot_setting, outpath=outpath)