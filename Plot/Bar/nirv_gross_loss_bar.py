import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
import scipy.sparse

def to_array(self):
    return self.toarray()

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba
import seaborn as sns
import GlobVars as gv

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)
# mpl.rcParams['axes.linewidth'] = 1.5
def cm2inch(value):
    return value/2.54

def main(dfs:list, grid:tuple, cols:list, plot_setting:dict, outpath:str):
    title_nums = ['a','b','c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(8), cm2inch(8))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes
            df, errors = dfs[i*ncols+j]
            xcol, ycol, ecol= cols[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            xlabel = plot_setting['xlabels'][i*ncols+j]
            ylabel = plot_setting['ylabels'][i*ncols+j]
            # ytick = plot_setting['yticks'][i*ncols+j]

            # Plot.
            df.plot(ax=ax, kind='bar', yerr=errors, capsize=0, color=['#f4b41a', '#299d8f',])
            ax.tick_params(axis='x', labelrotation=0)
            ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=2)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            # Plugins.
            ax.set_ylim(0, 70)
            ax.set_xlim(-0.6, 1.8)
            # ax.set_title(title_num+' '+title, loc='left', fontsize=LABEL_SIZE+1)
            ax.legend(loc='upper right', fontsize=LABEL_SIZE, frameon=False)

            # Remove frames and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=0.1, top=0.92, left=0.18, right=0.98, hspace=0, wspace=0.4)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    path = r"F:\Research\AMFEdge\EdgeVI\NIRvLossFrac_Monte_Amazon_Edge_2023.csv"
    df = pd.read_csv(path)
    cols = ['neg_edge_gain_frac', 'neg_int_gain_frac', 'pos_edge_gain_frac', 'pos_int_gain_frac', ]
    means = df[cols].mean() * 100
    stds = df[cols].std() * 100

    plt_df = pd.DataFrame({
        'Group': ['Loss', 'Loss','Gain', 'Gain'],
        'Feature': ['Edge', 'Interior']*2,'Contribution': means,
        'Contribution_std': stds})
    pivot = plt_df.pivot(index='Group', columns='Feature', values='Contribution')
    errors = plt_df.pivot(index='Group', columns='Feature', values='Contribution_std')

    dfs = [(pivot, errors)]

    grid = (1, 1)
    cols = [('Feature', 'Contribution', 'Contribution_std')] * 2
    titles = ['$R^2$=0.84', '$R^2$=0.78']

    plot_setting = {
        'titles': titles,
        'xlabels': [None],
        'ylabels': [r'Contribution (%)']*2,
    }

    outpath = r"E:\Thesis\AMFEdge\Figures\Description\nirv_gross_loss_bar.jpg"
    main(dfs, grid, cols, plot_setting, outpath)
