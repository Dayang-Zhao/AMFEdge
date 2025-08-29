import numpy as np
import pandas as pd

import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.gridspec as gridspec

VAR = 'fNIRv'
COUNT_COLUMN = VAR+'_count'
MEAN_COLUMN = VAR+'_mean'
MEDIAN_COLUMN = VAR+'_median'
STD_COLUMN = VAR+'_std'

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

def cal_95CI(row, var, grid):
    n = row[var+'_count_'+grid]
    mean = row[var+'_mean_'+grid]
    std_dev = row[var+'_stdDev_'+grid]
    
    # 计算标准误差
    se = std_dev / np.sqrt(n)
    
    # 查找 t 值
    t_value = stats.norm.ppf(0.975)  # 0.975 对应 95% 置信区间

    # 计算置信区间
    lower_bound = mean - t_value * se
    upper_bound = mean + t_value * se
    
    return pd.Series([lower_bound, upper_bound])


def main(dfs:list, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    # fig, axes = plt.subplots(
    #     nrows=nrows, ncols=ncols, 
    #     )
    fig = plt.figure()
    fig.set_size_inches(cm2inch(15), cm2inch(12))
    outer_gs = gridspec.GridSpec(2, 2)

    for i in range(nrows):
        for j in range(ncols):
            xlim = [0, 35]
            # 为每个大子图设置内部的2行1列子图（上下两个）
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1,             # 2行1列
                subplot_spec=outer_gs[i, j],
                hspace=0          # 子图间距为0，实现无缝拼接
            )
            df = dfs[i*ncols+j]

            # Plot setting.
            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1])
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]

            # Plot.
            ax1.errorbar(df['age'], df[VAR+'_mean_negrid'], 
                        yerr = df[VAR+'_mstd_swgrid'],
                        fmt='.', color='#3d98c2', label='negrids')

            ax2.errorbar(df['age'], df[VAR+'_mean_swgrid'], 
                        yerr = df[VAR+'_mstd_swgrid'],
                        fmt='.', color='#ed3e2e', label='swgrids')
            
            # Linear regression.
            result1 = stats.linregress(df['age'], df[VAR+'_mean_negrid'])
            x = np.linspace(xlim[0], xlim[1], 100)
            y1 = result1.intercept + result1.slope * x
            ax1.plot(x, y1, color='black', linewidth=1.5, linestyle='--', zorder=10)
            ax1.text(0.05, 0.2, '$r$= '+str(result1.rvalue.round(2)), fontsize=12, transform=ax1.transAxes)
            ax1.text(0.05, 0.08, '$p$= '+str(result1.pvalue.round(2)), fontsize=12, transform=ax1.transAxes)

            result2 = stats.linregress(df['age'], df[VAR+'_mean_swgrid'])
            x = np.linspace(xlim[0], xlim[1], 100)
            y2 = result2.intercept + result2.slope * x
            ax2.plot(x, y2, color='black', linewidth=1.5, linestyle='--', zorder=10)
            ax2.text(0.05, 0.6, '$r$= '+str(result2.rvalue.round(2)), fontsize=12, transform=ax2.transAxes)
            ax2.text(0.05, 0.48, '$p$= '+str(result2.pvalue.round(2)), fontsize=12, transform=ax2.transAxes)

            # if i == 0 and j ==0:
            #     ax1.legend(loc='best', frameon=False, prop={'size':10}, ncol=1)
                
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)
            ax1.set_ylabel('d$\Delta$NIRv (%)', labelpad=2)
            ax2.set_xlabel('Age (yr)', labelpad=2)
            # Y ticklabels is integer and their number is less than 6.
            ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
            ax1.minorticks_off()

            ax1.set_title(title_num+' '+title, loc='left')

    # Adjust.
    fig.subplots_adjust(bottom=0.08, top=0.95, left=0.10, right=0.97, hspace=0.35, wspace=0.25)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\EdgeAge\specEdge\anoVI_panAmazon_dspecUndistEdge_2023_age.xlsx"
    edge_types = ['grass', 'crop', 'water', 'nonveg']
    dfs = [pd.read_excel(path, sheet_name=edge_type) for edge_type in edge_types]

    # Plot setting.
    titles = ['Grass edge', 'Crop edge', 'Water edge', 'Non-veg edge']
    plot_setting = {'titles': titles}

    grid = (2, 2)
    outpath = r"E:\Thesis\AMFEdge\Figures\EdgeAge\pan_NIRv_specEdge.png"
    main(dfs=dfs, grid=grid, plot_setting=plot_setting, outpath=outpath)