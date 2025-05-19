import pandas as pd

import matplotlib.pyplot as plt
plt.ion()

def group_sum(df:pd.DataFrame, dist_thresh:int):
    sum_df = df.loc[df['Dist'] <= 3000].sum(axis=0, skipna=True)
    dst_df = df.loc[(df['Dist'] <= dist_thresh) & (df['Dist'] > 0)]
    dst_sum_df = dst_df.sum(axis=0, skipna=True)
    frac_df = dst_sum_df/sum_df
    frac_df = frac_df.drop(labels=['Id', 'Dist']).rename(lambda x: x.replace('_count', '_frac'), axis=0)
    sum_df = sum_df.drop(labels=['Id', 'Dist']).rename(lambda x: x.replace('_count', '_sum'), axis=0)
    dst_sum_df = dst_sum_df.drop(labels=['Id', 'Dist']).rename(lambda x: x.replace('_count', '_edgeSum'), axis=0)
    
    outdf = pd.concat([sum_df, dst_sum_df, frac_df], axis=0)
    return outdf

def dist_sum(df:pd.DataFrame, dist_thresh:int, dst_cols:list):
    dst_df = df.loc[:, ['Id', 'Dist']+dst_cols].copy()
    frac_df = dst_df.groupby('Id').apply(group_sum, dist_thresh=dist_thresh).reset_index()

    return frac_df

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\Edge_TreeCover\treeCover_Amazon_UndistEdge_2023.csv"
    df = pd.read_csv(path)
    dst_vars = ['undistForest', 'deforestLand', 'degradedForest', 'regrowthForest']
    dst_cols = [x+'_count' for x in dst_vars]
    
    dist = 300
    outdf = dist_sum(df, dist_thresh=dist, dst_cols=dst_cols)

    outpath = rf"F:\Research\AMFEdge\Edge_TreeCover\treeCover_Amazon_2023_within{dist}m.csv"
    outdf.to_csv(outpath, index=False)