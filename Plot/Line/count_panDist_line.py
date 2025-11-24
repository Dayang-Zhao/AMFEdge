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


def main(dfs:list, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True
        )
    fig.set_size_inches(cm2inch(9), cm2inch(9))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i]
            df = dfs[i*ncols+j]
            edge_df = df[df['Dist']>=60]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            color = plot_setting['colors'][i*ncols+j]
            label = plot_setting['labels'][i*ncols+j]
            ylim = plot_setting['ylims'][i*ncols+j]
            vline = plot_setting['vlines'][i*ncols+j]

            # Plot.
            ax.plot(edge_df['Dist']/1000, edge_df[COUNT_COLUMN]/1e7,
                    '.', color=color, linewidth=1.5, label=label, zorder=10)

            # Draw edge and intact forest area.
            ax.axvspan(0, vline, color= "#e5086b8f", alpha=0.2, lw=0)
            ax.set_xlim(-0.2, 6.2)
            ax.set_ylim(ylim)
            if i == nrows-1:
                ax.set_xlabel('$d$ (km)', fontsize=LABEL_SIZE+2, labelpad=-2)
            # Y ticklabels is integer and their number is less than 6.
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
            ax.minorticks_off()

            ax.text(0.85, 0.8, title, fontsize=LABEL_SIZE+1, transform=ax.transAxes)

    # Adjust.
    fig.text(0.01, 0.5, r'Number of pixels ($\times 10^7$)', va="center", rotation="vertical", fontsize=LABEL_SIZE+1)
    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.13, right=0.95, hspace=0, wspace=0.22)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Original data.
    path = r"F:\Research\AMFEdge\Edge\Main\anoVI_panAmazon_Edge_2023.xlsx"
    df1 = pd.read_excel(path, sheet_name='NE')
    df1 = df1.loc[df1['Dist']<=6000]
    df2 = pd.read_excel(path, sheet_name='SW')
    df2 = df2.loc[df2['Dist']<=6000]
    dfs = [df1, df2]

    # Plot setting.
    colors = ["#CC4348", "#3076CC"]
    labels = ['NE', 'SW']
    ylims = [(1,4), (2,5.5)]
    vlines = [1.253, 2.101]
    titles = ['NE', 'SW']
    plot_setting = {'colors': colors, 'labels': labels, 'ylims': ylims, 'vlines': vlines, 'titles': titles}

    grid = (2, 1)
    outpath = r"E:\Thesis\AMFEdge\Figures\Edge\pan_anoNIRv_count.jpg"

    main(dfs=dfs, grid=grid, plot_setting=plot_setting, outpath=outpath)