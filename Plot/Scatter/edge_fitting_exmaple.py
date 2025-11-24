import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import matplotlib as mpl
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

# Exponential Function
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def main(dfs:list, grid:tuple, cols:list, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(15), cm2inch(12))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i,j]
            edge_df, popt = dfs[i*ncols+j]
            xcol, ycol = cols[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            ylabel = plot_setting['ylabels'][i*ncols+j]
            xlim = plot_setting['xlims'][i*ncols+j]
            ylim = plot_setting['ylims'][i*ncols+j]

            # Plot.
            y = func(edge_df[xcol]*1000, *popt[0:3]).reset_index(drop=True)
            ax.plot(edge_df[xcol][1:], edge_df[ycol][1:], 'o', color='#275C9E', markersize=3, label='$\Delta$NIRv')
            ax.plot(edge_df[xcol], y, '-', color='#CC4348', )
            
            ax.axhline(y=y[0]+popt[3], color='#275C9E', linestyle='--', linewidth=1.5)
            ax.axvline(x=popt[4]/1000, color='#CC4348', linestyle='--', linewidth=1.5)


            ax.set_xlabel('Distance from edge (km)', fontsize=LABEL_SIZE)
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

            # Plugins.
            ax.set_title(title_num+' '+title, loc='left', fontsize=LABEL_SIZE+1)

            # Remove frames and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=0.1, top=0.92, left=0.12, right=0.98, hspace=0.4, wspace=0.25)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    VI_edge_path = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_Edge_2023.csv"
    VI_edge_df = pd.read_csv(VI_edge_path)
    VI_edge_df['Dist'] = VI_edge_df['Dist']/1000
    VI_edge_df['NIRv_mean'] = VI_edge_df['NIRv_mean']*-1
    VI_popt_path = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_Edge_Effect_2023.csv"
    VI_popt_df = pd.read_csv(VI_popt_path)
    VI_popt_df['nirv_para1'] = VI_popt_df['nirv_para1']*-1
    VI_popt_df['nirv_para3'] = VI_popt_df['nirv_para3']*-1
    VI_popt_df['nirv_magnitude'] = VI_popt_df['nirv_magnitude']*-1

    RH_edge_path = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_2023.csv"
    RH_edge_df = pd.read_csv(RH_edge_path)
    RH_edge_df['Dist'] = RH_edge_df['Dist']/1000
    RH_popt_path = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_Effect_2023.csv"
    RH_popt_df = pd.read_csv(RH_popt_path)

    ids = [776, 562] # Moderately and highly fragmented examples
    dfs = []
    for id in ids:
        dst_VI_edge_df = VI_edge_df.loc[(VI_edge_df['Id'] == id)&(VI_edge_df['Dist'] <= 6)&(VI_edge_df['Dist'] != -1),:]
        dst_VI_popt = VI_popt_df.loc[VI_popt_df['Id'] == id, ['nirv_para1', 'nirv_para2', 'nirv_para3', 'nirv_magnitude', 'nirv_scale']].values[0]
        dfs.append((dst_VI_edge_df, dst_VI_popt))
        print(dst_VI_popt)
        dst_RH_edge_df = RH_edge_df.loc[(RH_edge_df['Id'] == id)&(RH_edge_df['Dist'] <= 6)&(RH_edge_df['Dist'] != -1),:]
        dst_RH_popt = RH_popt_df.loc[RH_popt_df['Id'] == id, ['rh98_para1', 'rh98_para2', 'rh98_para3', 'rh98_magnitude', 'rh98_scale']].values[0]
        dfs.append((dst_RH_edge_df, dst_RH_popt))
        print(dst_RH_popt)

    grid = (2, 2)
    cols = [('Dist', 'NIRv_mean'), ('Dist', 'rh98_mean')]*2
    ylabels = [r'$\Delta$NIRv', 'RH98 (m)']*2
    titles = ['$\Delta$NIRv', 'RH98',
              'Bad Fit of $\Delta$NIRv', 'Bad Fit of RH98']
    xlims = [(-0.2,6.2)]*4
    ylims = [(-1,9), (10, 27), (-3,1), (20, 32)]
    plot_setting = {
        'titles': titles,
        'ylabels': ylabels,
        'xlims': xlims,
        'ylims': ylims,
    }
    outpath = r"E:\Thesis\AMFEdge\Figures\Model\edge_fitting_exmaple.pdf"

    main(dfs, grid, cols, plot_setting, outpath)