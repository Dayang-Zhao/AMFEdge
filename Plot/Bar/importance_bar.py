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

LABEL_SIZE = 10
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
    fig.set_size_inches(cm2inch(12), cm2inch(13))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            df = dfs[i*ncols+j]
            xcol, ycol, ecol= cols[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            xlabel = plot_setting['xlabels'][i*ncols+j]
            ylabel = plot_setting['ylabels'][i*ncols+j]
            # ytick = plot_setting['yticks'][i*ncols+j]

            # Plot.
            df.plot.barh(ax=ax, x=xcol, y=ycol,
                         edgecolor=None, facecolor=to_rgba("#299d8f", alpha=1),
                         linewidth=1.5,legend=False)
            ax.errorbar(
                x=df[ycol],              # x坐标：数值
                y=range(len(df)),          # y坐标：条的中心
                xerr=df[ecol],          # 横向误差
                fmt='none',                # 不画点，只画误差棒
                ecolor='black',            # 误差棒颜色
                capsize=2,                 # 两端小横线长度
                elinewidth=1.5              # 误差棒线宽
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

    fig.subplots_adjust(bottom=0.1, top=0.92, left=0.13, right=0.98, hspace=0, wspace=0.4)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    path = r"F:\Research\AMFEdge\Model\RF_PFI_importance.xlsx"
    mag_df = pd.read_excel(path, sheet_name='nirv_magnitude')
    scale_df = pd.read_excel(path, sheet_name='nirv_scale')
    yticks = dict(zip(
        ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
         'SCC_mean', 'sand_mean_mean',
        'MCWD_mean', 'surface_solar_radiation_downwards_sum_mean',
         'vpd_mean', 'total_precipitation_sum_mean','temperature_2m_mean'], 
        ['HAND','$S_{\Delta \mathrm{RH98}}$', '$M_{\Delta \mathrm{RH98}}$',
         'Soil Fertility', 'Soil Texture', 'MCWD',
        '$\Delta$PAR','$\Delta$VPD', '$\Delta P$', '$\Delta T$'
        ]))

    mag_df = mag_df.sort_values(by='Importance (ΔR²)', axis=0, ascending=True)
    mag_df['Feature'] = mag_df['Feature'].replace(yticks)
    scale_df = scale_df.sort_values(by='Importance (ΔR²)', axis=0, ascending=True)
    scale_df['Feature'] = scale_df['Feature'].replace(yticks)
    dfs = [mag_df, scale_df]

    grid = (1, 2)
    cols = [('Feature', 'Importance (ΔR²)', 'Importance std')] * 2
    titles = ['$R^2$=0.84', '$R^2$=0.81']

    plot_setting = {
        'titles': titles,
        'xlabels': [r'Importance for $M_{\nabla \mathrm{NIRv}}$ ($\Delta R^2$)', 
                    r'Importance for $S_{\nabla \mathrm{NIRv}}$ ($\Delta R^2$)'],
        'ylabels': [None]*2,
    }

    outpath = r"E:\Thesis\AMFEdge\Figures\Cause\nirv_RF_importance_bar.pdf"
    main(dfs, grid, cols, plot_setting, outpath)
