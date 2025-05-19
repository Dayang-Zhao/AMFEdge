import numpy as np
import pandas as pd

import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import seaborn as sns

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
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        )
    fig.set_size_inches(cm2inch(15), cm2inch(10))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i,j]
            df = dfs[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]

            # Plot.
            # ax.scatter(df['age'], df[VAR+'_mean_ngrid'], color='#3d98c2',marker='.', label='NGrids')
            ax.errorbar(df['age'], df[VAR+'_mean_ngrid'], 
                        yerr=[df[VAR+'_mean_ngrid'] - df[VAR+'_CI_lower_ngrid'],
                              df[VAR+'_CI_upper_ngrid'] - df[VAR+'_mean_ngrid']], 
                        fmt='.', color='#3d98c2', label='NGrids')
            # ax.scatter(df['age'], df[VAR+'_mean_pgrid'],color='#ed3e2e', marker='.', label='PGrids')
            ax.errorbar(df['age'], df[VAR+'_mean_pgrid'], 
                        yerr=[df[VAR+'_mean_pgrid'] - df[VAR+'_CI_lower_pgrid'],
                              df[VAR+'_CI_upper_pgrid'] - df[VAR+'_mean_pgrid']], 
                        fmt='.', color='#ed3e2e', label='PGrids')
            
            # ax.fill_between(edge_df1['Dist'], edge_df1['CI_lower'], edge_df1['CI_upper'], color='grey', alpha=0.5, label='95% CI')
            # ax.fill_between(edge_df2['Dist'], edge_df2['CI_lower'], edge_df2['CI_upper'], color='grey', alpha=0.5, label='95% CI')

            if i == 0 and j ==0:
                ax.legend(loc='best', frameon=False, prop={'size':10}, ncol=1)
                
            # plt.xlim(120, 200)
            ax.set_ylabel('d$\Delta$NIRv (%)')
            ax.set_xlabel('Age (yr)')
            # Y ticklabels is integer and their number is less than 6.
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
            ax.minorticks_off()

            ax.set_title(title_num+' '+title, loc='left')

    # Adjust.
    fig.subplots_adjust(bottom=0.11, top=0.92, left=0.10, right=0.97, hspace=0.55, wspace=0.25)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\EdgeAge\anoVI_panAmazon_dspecUndistEdge_2023_age.xlsx"
    edge_types = ['grass', 'crop', 'water', 'nonveg']
    dfs = [pd.read_excel(path, sheet_name=edge_type) for edge_type in edge_types]
    for df in dfs:
        df[[VAR+'_CI_lower_'+'ngrid', VAR+'_CI_upper_'+'ngrid']] = \
            df.apply(cal_95CI, args=(VAR, 'ngrid'), axis=1)
        df[[VAR+'_CI_lower_'+'pgrid', VAR+'_CI_upper_'+'pgrid']] = \
            df.apply(cal_95CI, args=(VAR, 'pgrid'), axis=1)

    # Plot setting.
    titles = ['Grass edge', 'Crop edge', 'Water edge', 'Non-veg edge']
    plot_setting = {'titles': titles}

    grid = (2, 2)
    outpath = r"E:\Thesis\AMFEdge\Figures\EdgeAge\pan_NIRv_specEdge.png"
    main(dfs=dfs, grid=grid, plot_setting=plot_setting, outpath=outpath)