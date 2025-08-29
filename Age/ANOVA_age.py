import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import pandas as pd
import numpy as np
import scipy.stats as stats

import GlobVars as gv
import matplotlib.pyplot as plt

def cal_edge_dmagnitude(edge_df, spec_edge_df, dst_vars):
    merge_df = spec_edge_df.merge(edge_df, on='Id', how='inner')
    for dst_var in dst_vars:
        intact_value = merge_df[dst_var.lower() + '_para3']
        merge_df['f'+dst_var+'_mean'] = (merge_df[dst_var + '_mean'] - intact_value)

    return merge_df

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\EdgeAge\sumEdge\anoVI_Amazon_sumUndistEdge_2023_age.csv"
    df = pd.read_csv(path)
    df = df[df['NIRv_count']>=600]
    dst_vars = ['NIRv', 'NDWI', 'EVI']

    # Calculate the difference in edge magnitude between specific edge and interior forest.
    edge_path = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
    edge_df = pd.read_csv(edge_path)
    fedge_df = cal_edge_dmagnitude(edge_df, df, dst_vars)
    fedge_df = fedge_df[['Id', 'age', 'fNIRv_mean', 'fNDWI_mean', 'fEVI_mean']]

    # Perform ANOVA for each variable
    ids = {'negrid':gv.NEGRID_IDS, 'swgrid':gv.SWGRID_IDS}
    for name, ids in ids.items():
        print(name)
        for dst_var in dst_vars:
            col = 'f' + dst_var + '_mean'
            anova_data = fedge_df[fedge_df['Id'].isin(ids)][['age', col]]
            anova_data = anova_data.dropna()

            # Perform ANOVA
            groups = [group[col].values for name, group in anova_data.groupby('age')]
            f_stat, p_value = stats.f_oneway(*groups)

            print(f"{dst_var} - F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
