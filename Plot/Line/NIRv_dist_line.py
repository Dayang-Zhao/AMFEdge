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
STD_COLUMN = VAR+'_stdDev'

LABEL_SIZE = 12
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

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


def main(df:pd.DataFrame, grid:tuple, dst_ids:list, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        )
    fig.set_size_inches(cm2inch(19), cm2inch(15))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            dst_id = dst_ids[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = 'Grid ' + str(dst_id)

            # Read target dataframe.
            dst_df = df[df['Id']==dst_id]
            edge_df = dst_df[dst_df['Dist']!=-1]
            intact_df = dst_df[dst_df['Dist']==-1]

            # Plot.
            # markers, caps, bars = ax.errorbar(edge_df['Dist'], edge_df['Mean'], yerr=edge_df['Std'],
            #             fmt='-', color='#3d98c2', ecolor='grey', linewidth=2, elinewidth=1.5, capsize=0, 
            #             label='Edge Forest')
            # [bar.set_alpha(0.3) for bar in bars]
            ax.scatter(edge_df['Dist'], edge_df[MEAN_COLUMN], color='#3d98c2')
            # ax.fill_between(edge_df['Dist'], edge_df['CI_lower'], edge_df['CI_upper'], color='grey', alpha=0.5, label='95% CI')

            # ax.axhline(y=intact_df[MEAN_COLUMN].values[0], color='#ED4043', linestyle='--', label='Intact Forest')

            if i == 0 and j ==0:
                ax.legend(loc='best', frameon=False, prop={'size':10}, ncol=1)

            # plt.xlim(120, 200)
            ax.set_ylabel('$\Delta$NIRv (%)')
            ax.set_xlabel('Distance (m)')
            # Y ticklabels is integer and their number is less than 6.
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
            ax.minorticks_off()

            ax.set_title(title_num+' '+title, loc='left')

    # Adjust.
    # fig.subplots_adjust(bottom=0.25, top=0.85, left=0.09, right=0.98, hspace=0.55, wspace=0.22)
    fig.subplots_adjust(bottom=0.1, top=0.92, left=0.09, right=0.98, hspace=0.4, wspace=0.25) # Plots: (2, 2)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\Edge\anoVI_Amazon_UndistEdge_2023.csv"
    df = pd.read_csv(path)
    df = df.loc[df['Dist']<=6000]
    df[['CI_lower', 'CI_upper']] = df.apply(cal_95CI, axis=1)

    grid = (2,2)
    # dst_ids = [160, 179, 279, 281]
    # dst_ids = [280, 300, 301, 340, ]
    dst_ids = [601, 602, 637, 638]
    outpath = None

    main(df=df, dst_ids=dst_ids, grid=grid, outpath=outpath)