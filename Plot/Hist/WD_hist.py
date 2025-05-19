import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns

LABEL_SIZE = 12
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

def main(data:pd.DataFrame, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True
        )
    fig.set_size_inches(cm2inch(15), cm2inch(18))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i]

            # Plot setting.
            xlabel = plot_setting['xlabels'][j+i*(ncols)]
            ylabel = plot_setting['ylabels'][j+i*(ncols)]
            title_num = title_nums[i*ncols+j]
            title = plot_setting['title'][j+i*(ncols)]

            # Read target dataframe.
            df = data[j+i*(ncols)]

            # Plot.
            ax.plot(df['Time'], df.iloc[:,1], '-', color='#1C6AB1')

            # Plot setting.
            ax.set_ylabel(ylabel=ylabel, labelpad=-0.2)
            # ax.set_xlabel(xlabel=xlabel, labelpad=-0.2)
            # if i==0 and j==0:
            #     ax.legend(
            #         loc='best', frameon=False, prop={'size':10}, ncol=1)
            # else:
            #     ax.legend([], [], frameon=False)
            ax.set_title(title_num+' '+title, loc='left')

    # Adjust.
    # fig.subplots_adjust(bottom=0.25, top=0.85, left=0.09, right=0.98, hspace=0.55, wspace=0.22)
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.12, right=0.96, hspace=0.3, wspace=0.1)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Data.
    path1 = r"F:\Research\TropicalForestEdge\Test\WD\WD.csv"
    df1 = pd.read_csv(path1, parse_dates=['Time'])
    path2 = r"F:\Research\TropicalForestEdge\Test\WD\CWD.csv"
    df2 = pd.read_csv(path2, parse_dates=['Time'])
    path3 = r"F:\Research\TropicalForestEdge\Test\WD\MCWD.csv"
    df3 = pd.read_csv(path3, parse_dates=['Time'])
    dfs = [df1, df2, df3]

    # Plot setting.
    grid = (3,1)
    xlabels = ['Time', 'Time', 'Time']
    ylabels = ['WD (mm)', 'CWD (mm)', 'MCWD (mm)']
    title = ['WD (mm)', 'CWD (mm)', 'MCWD (mm)']
    plot_setting = {'xlabels':xlabels, 'ylabels':ylabels, 'title':title}
    outpath = r"E:\Thesis\TropicalForestEdge\Figures\Test\WD_example.tif"

    main(data=dfs, grid=grid, plot_setting=plot_setting, outpath=outpath)