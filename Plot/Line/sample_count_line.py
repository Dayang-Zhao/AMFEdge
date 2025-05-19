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

def main(df:pd.DataFrame, grid:tuple, dst_ids:list, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        )
    fig.set_size_inches(cm2inch(15), cm2inch(8))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            dst_id = dst_ids[i*ncols+j]

            # Plot setting.
            xlabel = plot_setting['xlabels'][j+i*(ncols)]
            ylabel = plot_setting['ylabels'][j+i*(ncols)]
            title_num = title_nums[i*ncols+j]
            title = 'Grid ' + str(dst_id)

            # Read target dataframe.
            dst_df = df[df['Id']==dst_id]
            edge_df = dst_df[dst_df['Dist']!=-1]
            intact_df = dst_df[dst_df['Dist']==-1]

            # Plot.
            ax.plot(edge_df['Dist'], edge_df['NIRv_ano Count']/100000, color='#3d98c2')
            # ax.axhline(y=intact_df['NIRv_ano Count'].values[0], color='#ED4043', linestyle='--', label='Intact Forest')

            # Plot setting.
            ax.set_ylabel(ylabel=ylabel, labelpad=-0.2)
            ax.set_xlabel(xlabel=xlabel, labelpad=-0.2)
            if i==0 and j==0:
                ax.legend(
                    loc='best', frameon=False, prop={'size':10}, ncol=1)
            else:
                ax.legend([], [], frameon=False)
            ax.set_title(title_num+' '+title, loc='left')

    # Adjust.
    fig.subplots_adjust(bottom=0.15, top=0.92, left=0.09, right=0.96, hspace=0.55, wspace=0.3)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    path = r"F:\Research\TropicalForestEdge\Test\NIRv\Hist\Grid_Hist_sum.csv"
    df = pd.read_csv(path)
    df2 = df.groupby(['Id', 'Dist']).sum().reset_index()
    grid = (1,2)
    dst_ids = [280, 300]
    xlabels = ['Distance (m)', 'Distance (m)']
    ylabels = [r'Count ($\times 10^5$)', r'Count ($\times 10^5$)']
    plot_setting = {'xlabels':xlabels, 'ylabels':ylabels}
    outpath = r"E:\Thesis\TropicalForestEdge\Figures\Test\sample_count_line.tif"

    main(df=df2, grid=grid, dst_ids=dst_ids, plot_setting=plot_setting, outpath=outpath)