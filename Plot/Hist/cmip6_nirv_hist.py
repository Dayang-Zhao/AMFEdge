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
# outpath = r"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_pred_ave.csv"
# ave_df = pd.read_csv(outpath)

def cm2inch(value):
    return value/2.54

def main(dfs:list, col:str, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True
        )
    fig.set_size_inches(cm2inch(10), cm2inch(7))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i]

            # Plot setting.
            data = dfs[i]
            palette = plot_setting['palette']
            xlabel = plot_setting['xlabel']
            ylabel = plot_setting['ylabel']
            label = plot_setting['labels'][i]

            # Plot.
            sns.kdeplot(data=data, x=col, hue='scenario', ax=ax, palette=palette, legend=False,
                        common_norm=False, linewidth=2, fill=True, alpha=0.5)
            # sns.histplot(data=data, x=col, hue='scenario', ax=ax, palette=palette, legend=False, bins=30)
            
            # Average line.
            # scenario = data['scenario'].values[0]
            # pos_ave = ave_df.loc[ave_df['scenario']==scenario, 'pos_ave'].values[0]
            # neg_ave = ave_df.loc[ave_df['scenario']==scenario, 'neg_ave'].values[0]
            # ax.axvline(x=pos_ave, color=palette[scenario], linestyle='-.', linewidth=2)
            # ax.axvline(x=neg_ave, color=palette[scenario], linestyle='-.', linewidth=2)

            # Text.
            # ax.text(0.8, 0.4, f'{pos_ave:.2f}%', transform=ax.transAxes,
            #         ha='right', color=palette[scenario], fontsize=LABEL_SIZE-2)
            # ax.text(0.15, 0.4, f'{neg_ave:.2f}%', transform=ax.transAxes,
            #         ha='right', color=palette[scenario], fontsize=LABEL_SIZE-2)

            # Plot setting.
            ax.set_xlabel(xlabel=xlabel, labelpad=-0.2)
            ax.set_ylabel(ylabel=None)

            # Legend
            ax.legend(labels=[label], loc="upper right", frameon=False, fontsize=LABEL_SIZE-2)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2)

            # Remove frames and ticks
            ax.set_yticks([])       
            ax.set_yticklabels([]) 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Adjust.
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.96, hspace=0, wspace=0.1)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Data.
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']
    dfs = []
    for scenario in scenarios:
        df = pd.read_csv(rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_Edge_pred_{scenario}.csv")
        df.drop(columns=['model'], inplace=True)
        df = df.groupby(['Id', 'year']).mean().reset_index()
        df['scenario'] = scenario
        dfs.append(df)
    # data = pd.concat(dfs, ignore_index=True)
    # data = data.sort_values(by='scenario')
    col = 'nirv_magnitude'

    # Plot setting.
    grid = (3,1)
    xlabel = r'$M_{\Delta \mathrm{NIRv}}$ (%)'
    ylabel = 'Probability Density'
    labels = ['RCP 2.6', 'RCP 4.5', 'RCP 8.5']
    title = ['']
    palette = dict(zip(scenarios, ['#576fa0', '#e3b87f','#b57979']))
    plot_setting = {'xlabel':xlabel, 'ylabel':ylabel, 'title':title, 
                    'palette':palette, 'labels':labels}
    outpath = r"E:\Thesis\AMFEdge\Figures\CMIP6\nirv_prj_kde.pdf"

    main(dfs=dfs, col=col,grid=grid, plot_setting=plot_setting, outpath=outpath)