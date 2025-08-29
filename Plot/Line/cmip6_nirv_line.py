import numpy as np
import pandas as pd
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.ticker import MaxNLocator
import seaborn as sns

LABEL_SIZE = 12
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

def main(dfs:list, col:str, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True
        )
    fig.set_size_inches(cm2inch(10.5), cm2inch(7))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i]

            # Plot setting.
            data = dfs[i]
            palette = plot_setting['palette']
            xlabel = plot_setting['xlabel']
            ylabel = plot_setting['ylabel']
            ylim = plot_setting['ylim'][i]
            label = plot_setting['labels'][i]

            # Plot.
            sns.lineplot(data=data, x='year', y=col, hue='scenario',
                        ax=ax, palette=palette, legend=False,
                        linewidth=2)
            for s in data['scenario'].unique():
                sub = data[data["scenario"] == s]
                ax.fill_between(
                    sub["year"], 
                    sub["ci_lower"], 
                    sub["ci_upper"], 
                    color=palette[s], alpha=0.2
                )

            # Plot setting.
            ax.set_xlabel(xlabel=xlabel, labelpad=-0.2)
            ax.set_ylabel(ylabel=None)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

            # Legend
            # ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax.set_ylim(ylim)

            # Remove frames and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Adjust.
    fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical")
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.96, hspace=0, wspace=0.1)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Data.
    col = 'nirv_magnitude'
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']
    def cal_ave2std(df):
        ave_df = df.groupby('year').mean().reset_index()

        # Smooth.
        ave_df[col] = ave_df[col].rolling(window=5, center=True).mean()

        def ci_95(x):
            n = len(x)
            mean = np.mean(x)
            sem = stats.sem(x)
            h = sem * stats.t.ppf((1 + 0.95)/2., n-1)
            return pd.Series([mean, mean-h, mean+h], index=["mean","ci_lower","ci_upper"])
        std_df = df.groupby('year')[col].apply(ci_95).unstack().reset_index()
        outdf = pd.merge(ave_df, std_df, on='year', suffixes=('_mean', '_std'))
        return outdf
    
    posdfs, negdfs = [], []
    for scenario in scenarios:
        df = pd.read_csv(rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_pred_{scenario}.csv")
        posdf = df[df['nirv_magnitude'] > 0]
        negdf = df[df['nirv_magnitude'] < 0]
        out_posdf = cal_ave2std(posdf) 
        out_negdf = cal_ave2std(negdf)
        out_posdf['scenario'] = scenario
        out_negdf['scenario'] = scenario
        posdfs.append(out_posdf)
        negdfs.append(out_negdf)
    posdf = pd.concat(posdfs, ignore_index=True)
    negdf = pd.concat(negdfs, ignore_index=True)
    dfs = [posdf, negdf]

    # Plot setting.
    grid = (2,1)
    ylabel = r'$M_{\nabla \mathrm{NIRv}}$ (%)'
    xlabel = 'Year'
    labels = ['RCP 2.6', 'RCP 4.5', 'RCP 8.5']
    ylims = [(0.7,1.5), (-2.5,-0.9)]
    title = ['']
    palette = dict(zip(scenarios, ['#576fa0', '#e3b87f','#b57979']))
    plot_setting = {'xlabel':xlabel, 'ylabel':ylabel, 'title':title, 
                    'palette':palette, 'labels':labels, 'ylim':ylims}
    outpath = r"E:\Thesis\AMFEdge\Figures\CMIP6\nirv_prj_line.pdf"

    main(dfs=dfs, col=col,grid=grid, plot_setting=plot_setting, outpath=outpath)