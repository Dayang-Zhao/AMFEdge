import numpy as np
import pandas as pd

import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns

VAR = 'NIRv'
COUNT_COLUMN = VAR+'_count'
MEAN_COLUMN = VAR+'_mean'
MEDIAN_COLUMN = VAR+'_median'
STD_COLUMN = VAR+'_std'

LABEL_SIZE = 12
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)
# mpl.rcParams['axes.linewidth'] = 1
def cm2inch(value):
    return value/2.54

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def cal_95CI(row):
    n = row[COUNT_COLUMN]
    mean = row[MEAN_COLUMN]
    std_dev = row[STD_COLUMN]
    
    # 计算标准误差
    se = std_dev / np.sqrt(n)
    
    # 查找 t 值
    t_value = stats.norm.ppf(0.975)  # 0.975 对应 95% 置信区间

    # 计算置信区间
    lower_bound = mean - t_value * se
    upper_bound = mean + t_value * se
    
    return pd.Series([lower_bound, upper_bound])


def main(dfs:list, fit_dfs:list, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True
        )
    fig.set_size_inches(cm2inch(13), cm2inch(6.5))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            df = dfs[i*ncols+j]
            fit_df = fit_dfs[i*ncols+j]
            edge_df = df[df['Dist']>=60]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = 'Basin-wide'
            color = plot_setting['colors'][i*ncols+j]
            label = plot_setting['labels'][i*ncols+j]
            ylim = plot_setting['ylims'][i*ncols+j]

            # Plot.
            markers, caps, bars = ax.errorbar(
                edge_df['Dist']/1000, edge_df[MEAN_COLUMN], 
                yerr=edge_df[VAR+'_mstd']*0.5,
                fmt='.', markersize=10, color="#275C9E", ecolor='grey', elinewidth=1)
            bars[0].set_alpha(0.4)
            # ax.plot(edge_df['Dist']/1000, func(edge_df['Dist'], *fit_df.iloc[0, 0:3]),
            #         '-', color=color, linewidth=1.5, label=label, zorder=10)

            # ax.legend(loc='best', frameon=False, prop={'size':10}, ncol=1)

            ax.set_ylim(ylim)
            ax.set_ylabel(label+' NIRv (-)')
            # if i == nrows-1:
            ax.set_xlabel('$d$ (km)')
            # Y ticklabels is integer and their number is less than 6.
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
            ax.minorticks_off()

            # for spine in ax.spines.values():
            #     spine.set_linewidth(2)

            # if i == 0 and j ==0:
            ax.set_title(title_num, loc='left')

    # Adjust.
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.13, right=0.97, hspace=0, wspace=0.4)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Original data.
    path = r"F:\Research\AMFEdge\EdgeVI\VI_panAmazon_UndistEdge_2023.xlsx"
    df1 = pd.read_excel(path, sheet_name='NEGRID')
    df1 = df1.loc[df1['Dist']<=6000]
    df2 = pd.read_excel(path, sheet_name='SWGRID')
    df2 = df2.loc[df2['Dist']<=6000]
    dfs = [df1, df2]

    # Fitting data.
    path = r"F:\Research\AMFEdge\EdgeVI\VI_panAmazon_UndistEdge_effect_2023.xlsx"
    df1 = pd.read_excel(path, sheet_name='NEGRID')
    df2 = pd.read_excel(path, sheet_name='SWGRID')
    fit_dfs = [df1, df2]

    # Plot setting.
    colors = ["#CC4348", "#3076CC"]
    labels = ['NE', 'SW']
    # ylims = [(-2.1, 3), (-0.5, 2.8)]
    ylims = [(0.2, 0.26), (0.2, 0.26)]
    plot_setting = {'colors': colors, 'labels': labels, 'ylims': ylims}

    grid = (1, 2)
    outpath = r"E:\Thesis\AMFEdge\Figures\EdgeVI\pan_NIRv_edge.pdf"

    main(dfs=dfs, fit_dfs=fit_dfs, grid=grid, plot_setting=plot_setting, outpath=outpath)