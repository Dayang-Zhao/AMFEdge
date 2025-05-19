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
    
    # 总体方差与标准差
    total_variance = (within_var + between_var) / N
    total_std = np.sqrt(total_variance)
    
    return N, weighted_mean, total_std

def cal_edge_dmagnitude(edge_df, spec_edge_df, dst_vars):
    merge_df = spec_edge_df.merge(edge_df, on='Id', how='inner')
    for dst_var in dst_vars:
        intact_value = 0.1*merge_df[dst_var.lower() + '_para1'] + merge_df[dst_var.lower() + '_para3']
        merge_df['f'+dst_var+'_mean'] = (merge_df[dst_var + '_mean'] - intact_value)
            # *100/merge_df[dst_var.lower() + '_magnitude']
        merge_df['f'+dst_var+'_count'] = merge_df[dst_var+'_count']
        merge_df['f'+dst_var+'_stdDev'] = merge_df[dst_var+'_stdDev']
            # *100/merge_df[dst_var.lower() + '_magnitude']

    return merge_df

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\EdgeAge\anoVI_Amazon_specUndistEdge_2023_age.xlsx"
    edge_type = 'grass'
    print(edge_type)
    df = pd.read_excel(path, sheet_name=edge_type)
    df = df[df['NIRv_count']>=600]
    dst_vars = ['NIRv', 'NDWI', 'EVI']
    ids = {'pgrid':gv.PGRID_IDS, 'ngrid':gv.NGRID_IDS}

    # Calculate the difference in edge magnitude between specific edge and actual edge.
    edge_path = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
    edge_df = pd.read_csv(edge_path)
    fedge_df = cal_edge_dmagnitude(edge_df, df, dst_vars)

    dst_vars = ['fNIRv', 'fNDWI', 'fEVI']
    outdfs = []
    for name, id in ids.items():
        dst_df = fedge_df[fedge_df['Id'].isin(id)]
        results = []
        for age, group in dst_df.groupby('age'):
            outrow = {'age': age}
            for dst_var in dst_vars:
                count, mean, std = estimate_population_mean_std(group, dst_var)
                outrow[dst_var + '_count'] = count
                outrow[dst_var + '_mean'] = mean
                outrow[dst_var + '_stdDev'] = std
            results.append(outrow)
        outdf = pd.DataFrame(results)
        outdf = outdf.rename(columns={col: col + '_' + name for col in outdf.columns if col != 'age'})
        outdfs.append(outdf)
    outdf = outdfs[0].merge(outdfs[1], on='age', how='inner')

    # Save the output dataframe to a CSV file.
    outpath = r"F:\Research\AMFEdge\EdgeAge\anoVI_panAmazon_dspecUndistEdge_2023_age.xlsx"
    sd.save_pd_as_excel(outdf, outpath, sheet_name=edge_type, index=False, add_row_or_col='col')