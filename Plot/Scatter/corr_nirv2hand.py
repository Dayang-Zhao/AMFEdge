import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import linregress
import Attribution.LcoRF as lcorf

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

def main(df:pd.DataFrame, cols:list, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(13), cm2inch(6.5))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            xcol, ycol = cols[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            xlabel = plot_setting['xlabels'][i*ncols+j]
            ylabel = plot_setting['ylabels'][i*ncols+j]
            xlim = plot_setting['xlims'][i*ncols+j]
            ylim = plot_setting['ylims'][i*ncols+j]

            # Plot.
            sc = ax.scatter(
                df[xcol],df[ycol], c='#299d8f', s=40,
                marker='o', edgecolors='black', alpha=0.8,
            )
            result = linregress(df[xcol], df[ycol])
            x = np.linspace(xlim[0], xlim[1], 100)
            y = result.intercept + result.slope * x
            ax.plot(x, y, color='black', linewidth=1.5, linestyle='--')
            ax.text(0.6, 0.1, '***$r$= '+str(result.rvalue.round(2)), fontsize=12, transform=ax.transAxes)
        
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
            ax.minorticks_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabel, labelpad=-3)
            ax.set_ylabel(ylabel)
            # Remove frames and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_title(f"{title_num} {title}", fontsize=LABEL_SIZE+2, loc='left')
    
    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.98, hspace=0, wspace=0.3)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    # Train the model.
    xcols = ['rh98_magnitude','rh98_scale']
    ycol = 'nirv_magnitude'

    path = r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution.csv"
    df = pd.read_csv(path)
    df['rh98_magnitude'] = df['rh98_magnitude']*-1
    df['rh98_scale'] = df['rh98_scale']/1000
    df = df[(df['nirv_scale'] <= 6000)]
    df = df.dropna(subset=xcols+[ycol])

    # Plot
    df = df[(df['rh98_scale'] <= 6)]
    grid = (1,2)
    cols = [('rh98_magnitude', 'nirv_magnitude'),('rh98_scale', 'nirv_magnitude')]

    plot_setting = {
        'titles': ['', ''],
        'xlabels': ['$M_{\mathrm{RH98}}$ (m)', '$S_{\mathrm{RH98}}$ (km)']*2,
        'ylabels': [r'$M_{\nabla \mathrm{NIRv}}$ (%)']*2,
        'xlims': [[0, 10], [0, 6]],
        'ylims': [[-6, 8], [-6, 8]],
    }
    outpath = r'E:\Thesis\AMFEdge\Figures\Cause\Mnirv_RH98.pdf'
    main(df, cols, grid, plot_setting, outpath=outpath)
