import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce

from sklearn.preprocessing import StandardScaler
import scipy.sparse

def to_array(self):
    return self.toarray()

scipy.sparse.spmatrix.A = property(to_array)
from pygam import LinearGAM, s, f, te

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba
import seaborn as sns
import GlobVars as gv

LABEL_SIZE = 12
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
def cm2inch(value):
    return value/2.54

def main(dfs:list, grid:tuple, cols:list, plot_setting:dict, outpath:str):
    title_nums = ['a','b','c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(6), cm2inch(10))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            df = dfs[i*ncols+j]
            xcol, ycol, hcol = cols[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            xlabel = plot_setting['xlabels'][i*ncols+j]
            ylabel = plot_setting['ylabels'][i*ncols+j]
            # ytick = plot_setting['yticks'][i*ncols+j]

            # Plot.
            sns.histplot(
                data=df, x=xcol, weights=ycol,
                hue=hcol, ax=ax, multiple='stack',
                palette=dict(zip(['Drought', 'Forest structure', 'Hydrology', 'Soil'],
                                 ["#CC4348", '#299d8f', '#3076CC', '#f4b41a'])),
                edgecolor='none'
            )
            # ax.set_yticks(range(len(df[xcol])), ytick,rotation=0)
            ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

            ax.xaxis.set_tick_params(width=1.5)
            ax.yaxis.set_tick_params(width=1.5)

            # Plugins.
            ax.set_title(title_num+' '+title, loc='left', fontsize=LABEL_SIZE+1)
            ax.legend().remove()

            # Remove frames and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=0.08, top=0.92, left=0.13, right=0.98, hspace=0, wspace=0.9)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    path = r"F:\Research\AMFEdge\Model\RF_Edge_PFI_importance.xlsx"
    mag_df = pd.read_excel(path, sheet_name='nirv_magnitude')
    scale_df = pd.read_excel(path, sheet_name='nirv_scale')
    mag_df = mag_df.sort_values(by='Importance (ΔR²)', axis=0, ascending=True)
    mag_df['Variable'] = 'Variable'
    scale_df = scale_df.sort_values(by='Importance (ΔR²)', axis=0, ascending=True)
    scale_df['Variable'] = 'Variable'
    order = ['Drought', 'Forest structure', 'Hydrology', 'Soil']

    # Convert 'Group' to categorical with desired order
    mag_df['Group'] = pd.Categorical(mag_df['Group'], categories=order, ordered=True)
    mag_df = mag_df.sort_values(by='Group')
    scale_df['Group'] = pd.Categorical(scale_df['Group'], categories=order, ordered=True)
    scale_df = scale_df.sort_values(by='Group')
    dfs = [mag_df, scale_df]

    grid = (1, 2)
    cols = [('Variable', 'Importance percent', 'Group')] * 2
    titles = ['','']

    plot_setting = {
        'titles': titles,
        'ylabels': [r'Variance explained in the model (%)', 
                    r'Variance explained in the model (%)'],
        'xlabels': [None]*2,
    }

    outpath = r"E:\Thesis\AMFEdge\Figures\Cause\nirv_RF_contribution_hist.pdf"
    main(dfs, grid, cols, plot_setting, outpath)
