import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import numpy as np
import pandas as pd

import GlobVars as gv
import Data.save_data as sd

import matplotlib.pyplot as plt
plt.ion()

def estimate_population_mean_std(df, dst_var:str):
    """
    根据各组的样本数、均值和标准差，估计总体均值和标准差。

    参数:
        df: pd.DataFrame
            包含每组统计信息的 DataFrame。
        n_col: str
            样本数的列名（默认为 'n'）。
        mean_col: str
            均值的列名（默认为 'mean'）。
        std_col: str
            标准差的列名（默认为 'std'）。

    返回:
        (population_mean, population_std)
    """
    n_col, mean_col, std_col = dst_var + '_count', dst_var + '_mean', dst_var + '_stdDev'
    n = df[n_col]
    mean = df[mean_col]
    std = df[std_col]
    N = n.sum()
    
    # 加权总体均值
    weighted_mean = (n * mean).sum() / N
    
    # 组内方差部分
    within_var = ((n - 1) * std**2).sum()
    
    # 组间方差部分
    between_var = (n * (mean - weighted_mean)**2).sum()

    # 均值标准差
    mstd = mean.std()
    
    # 总体方差与标准差
    total_variance = (within_var + between_var) / N
    total_std = np.sqrt(total_variance)
    
    return N, weighted_mean, mstd, total_std

if __name__ == '__main__':
    # Obtain the ids with valid edge effect curves.
    edge_path = r"F:\Research\AMFEdge\EdgeVI\VI_Amazon_UndistEdge_Effect_2023.csv"
    edge_df = pd.read_csv(edge_path)
    nonan_ids = edge_df.loc[edge_df['nirv_scale'].notna(), 'Id']
    path = r"F:\Research\AMFEdge\EdgeVI\VI_Amazon_UndistEdge_2023.csv"
    df = pd.read_csv(path)
    df = df[df['Id'].isin(gv.NEGRID_IDS) & df['Id'].isin(nonan_ids)]
    dst_vars = ['NIRv', 'NDWI', 'EVI']

    outdfs = []
    for dist in gv.DISTS:
        dst_df = df.loc[df['Dist'] == dist]
        outrow = {'Dist': dist}
        for dst_var in dst_vars:
            count, mean, mstd, std = estimate_population_mean_std(dst_df, dst_var)
            outrow[dst_var + '_count'] = count
            outrow[dst_var + '_mean'] = mean
            outrow[dst_var + '_mstd'] = mstd
            outrow[dst_var + '_std'] = std
        outdf = pd.DataFrame(outrow, index=[0])
        outdfs.append(outdf)

    outdf = pd.concat(outdfs, axis=0, ignore_index=True)

    # Save the output dataframe to a CSV file.
    outdf = outdf[outdf['Dist']<=6300]
    outpath = r"F:\Research\AMFEdge\EdgeVI\VI_panAmazon_UndistEdge_2023.xlsx"
    sd.save_pd_as_excel(outdf, outpath, sheet_name='NEGRID', index=False, add_row_or_col='col')