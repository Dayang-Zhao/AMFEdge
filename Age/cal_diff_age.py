import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import pandas as pd
import numpy as np
import scipy.stats as stats
from functools import reduce

import GlobVars as gv
import matplotlib.pyplot as plt

def cal_edge_dmagnitude(edge_df, spec_edge_df, dst_vars):
    merge_df = spec_edge_df.merge(edge_df, on='Id', how='inner')

    outdfs = []
    for dst_var in dst_vars:
        dst_merge_df = merge_df[merge_df[dst_var.lower()+'_r2']>=0.8].copy().reset_index(drop=True)
        dst_merge_df['intact_' + dst_var.lower()] = dst_merge_df[dst_var.lower() + '_para3']
        dst_merge_df['d'+dst_var+'_mean'] = (dst_merge_df[dst_var + '_mean'] - dst_merge_df['intact_' + dst_var.lower()])
        outdfs.append(dst_merge_df[['Id', 'age', dst_var+'_mean', 'intact_' + dst_var.lower(), 'd'+dst_var+'_mean']])

    outdf = reduce(lambda left, right: pd.merge(left, right, on=['Id','age'], how='outer'), outdfs)
    return outdf

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\EdgeAge\anoVI_Amazon_sumUndistEdge_2023_age.csv"
    df = pd.read_csv(path)
    df = df[df['NIRv_count']>=600]
    dst_vars = ['NIRv', 'NDWI', 'EVI']

    # Calculate the difference in edge magnitude between specific edge and interior forest.
    edge_path = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_Edge_Effect_2023.csv"
    edge_df = pd.read_csv(edge_path)
    dedge_df = cal_edge_dmagnitude(edge_df, df, dst_vars)
    # dedge_df = dedge_df[['Id', 'age', 'dNIRv_mean', 'dNDWI_mean', 'dEVI_mean']]

    # Save.
    outpath = r"F:\Research\AMFEdge\EdgeAge\anoVI_Amazon_sumUndistEdge_effect_2023_age.csv"
    dedge_df.to_csv(outpath, index=False)

