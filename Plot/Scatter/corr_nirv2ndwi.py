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
            xcol, ycol, ccol = cols[i*ncols+j]

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
            df.plot.scatter(ax=ax, x=xcol, y=ycol, c=ccol, cmap=cmap, norm=norm, colorbar=False)
            ## Plot regression line.
            x = np.linspace(xlim[0], xlim[1], 100)
            y = result.intercept + result.slope * x
            ax.plot(x, y, color='black', linewidth=1.5, linestyle='--')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

            # Plugins.
            ax.set_title(title_num+' '+title, loc='left', fontsize=LABEL_SIZE+1)
            ax.text(0.05, 0.90, f"$r$ = {result.rvalue:.2f}", transform=ax.transAxes, fontsize=LABEL_SIZE+2)
            ax.text(0.05, 0.80, f"$p$ < 0.001", transform=ax.transAxes, fontsize=LABEL_SIZE+2)

    fig.subplots_adjust(bottom=0.2, top=0.92, left=0.12, right=0.98, hspace=0, wspace=0.25)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    path = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
    df = pd.read_csv(path)
    df['ndwi_scale'] = df['ndwi_scale']/1000
    df['nirv_scale'] = df['nirv_scale']/1000
    dst_df = df.loc[
        (df['nirv_scale']!=6.001)&(df['ndwi_scale']!=6.001)&
                    (~np.isnan(df['nirv_scale']))&(~np.isnan(df['ndwi_scale']))]
    dfs = [dst_df]*2
    cols = [
        ('ndwi_magnitude','nirv_magnitude',  'nirv_magnitude'),
        ('ndwi_scale', 'nirv_scale', 'nirv_magnitude')
    ]
    
    # Plot setting.
    cmap2 = mpl.cm.RdBu_r
    levels = np.arange(-8, 9, 2)
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)
    cmaps = [cmap2, cmap2]
    norms = [norm2, norm2]
    ylabels = ['$\Delta$NIRv Magnitude (%)', '$\Delta$NIRv Scale (km)']
    xlabels = ['$\Delta$NDWI Magnitude (%)', '$\Delta$NDWI Scale (km)']
    xlims = [(-22,8), (0, 6.5)]
    ylims = [(-11, 7), (0, 6.5)]
    extend = ['both', 'max']
    titles = ['', '']

    plot_setting = {'cmaps': cmaps,'norms': norms,'extends': extend,'titles': titles,
                    'xlabels': xlabels, 'ylabels': ylabels, 'xlims': xlims, 'ylims': ylims}
    outpath = "E:\Thesis\AMFEdge\Figures\Cause\corr_nirv2ndwi.pdf"
    main(dfs, (1, 2), cols, plot_setting, outpath=outpath)