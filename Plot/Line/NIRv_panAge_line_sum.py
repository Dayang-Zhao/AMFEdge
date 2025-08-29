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
            xlim = [0, 35]
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            # Add text.
            def _add_rp(linear_regression_result):
                r = linear_regression_result.slope
                p = linear_regression_result.pvalue
                if p<0.001:
                    return f'***slope ={r.round(3)}'
                elif p<0.01:
                    return f'**slop$ ={r.round(3)}'
                elif p<0.05:
                    return f'*slope ={r.round(3)}'
                else:
                    return f'slope ={r.round(3)}'

            # Plot.
            ax.errorbar(df['age'], df[VAR+'_mean_grid']*-1,  yerr = df[VAR+'_mstd_grid'],
                        fmt='.', markersize=8, color='#ed3e2e', ecolor='#3d98c2', elinewidth=1)

            # Linear regression for the first five years.
            if i == 1:
                result = stats.linregress(df['age'][:5], df[VAR+'_mean_grid'][:5])
                x = np.linspace(xlim[0], xlim[0]+20, 100)
                y1 = result.intercept + result.slope * x
                ax.plot(x, y1, color="#D458D4", linewidth=2, linestyle='--', zorder=10)
                ax.text(0.55, 0.85, _add_rp(result), fontsize=12, color="#D458D4", transform=ax.transAxes)

            # Linear regression for the entire.
            result = stats.linregress(df['age'], df[VAR+'_mean_grid']*-1)
            x = np.linspace(xlim[0], xlim[1], 100)
            y1 = result.intercept + result.slope * x
            ax.plot(x, y1, color='black', linewidth=1.5, linestyle='--', zorder=10)

            if i == 0:
                text_y = 0.8
            else:
                text_y = 0.1
            ax.text(0.05, text_y, _add_rp(result), fontsize=12, transform=ax.transAxes)
            # ax.text(0.05, 0.08, '$p$= '+str(result.pvalue.round(2)), fontsize=12, transform=ax.transAxes)
            
            # if i == 0 and j ==0:
            #     ax.legend(loc='best', frameon=False, prop={'size':10}, ncol=1)
                
            ax.set_xlim(xlim)
            ax.set_ylabel('$\Delta$NIRv Magnitude(%)')
            ax.set_xlabel('Time since edge creation (yr)')
            # Y ticklabels is integer and their number is less than 6.
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
            ax.minorticks_off()

            ax.set_title(title_num+' '+title, loc='left')

    # Adjust.
    fig.subplots_adjust(bottom=0.11, top=0.92, left=0.10, right=0.97, hspace=0.55, wspace=0.25)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\EdgeAge\anoVI_panAmazon_dsumUndistEdge_2023_age_sum.csv"
    edge_types = ['grass', 'crop', 'water', 'nonveg']
    dfs = [pd.read_csv(path) for edge_type in edge_types]

    # Plot setting.
    titles = ['Grass edge', 'Crop edge', 'Water edge', 'Barren edge']
    plot_setting = {'titles': titles}

    grid = (2, 2)
    outpath = r"E:\Thesis\AMFEdge\Figures\EdgeAge\pan_NIRv_sumEdge.pdf"
    main(dfs=dfs, grid=grid, plot_setting=plot_setting, outpath=outpath)