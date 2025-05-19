import pandas as pd

import matplotlib.pyplot as plt
plt.ion()

def group_frac(df:pd.DataFrame, frac_thresh:float, dst_cols:list):
    sum_df = df.loc[df['Dist'] <= 3000, dst_cols].sum(axis=0, skipna=True) # Intact + <=3000m
    csum_df = df.loc[df['Dist'] > 0, dst_cols].cumsum(axis=0, skipna=True)
    frac_df = csum_df*100/sum_df
    frac_df[['Dist']] = df.loc[df['Dist'] > 0, ['Dist']]
    
    outSeries = pd.Series()
    for dst_col in dst_cols:
        try:
            dist = frac_df[frac_df[dst_col] <= frac_thresh]['Dist'].iloc[-1]
        except Exception:
            dist = frac_df['Dist'].iloc[0]
        outSeries[dst_col.replace('_count', '_dist')] = dist
    
    return outSeries

def dist_sum(df:pd.DataFrame, frac_thresh:float, dst_cols:list):
    dst_df = df.loc[:, ['Id', 'Dist']+dst_cols].copy()
    dist_df = dst_df.groupby('Id').apply(
        group_frac, frac_thresh=frac_thresh, dst_cols=dst_cols
    ).reset_index()

    return dist_df

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\Edge_TreeCover\treeCover_Amazon_UndistEdge_2023.csv"
    df = pd.read_csv(path)
    dst_vars = ['undistForest', 'deforestLand', 'degradedForest', 'regrowthForest']
    dst_cols = [x+'_count' for x in dst_vars]
    
    frac = 95
    outdf = dist_sum(df, frac_thresh=frac, dst_cols=dst_cols)

    outpath = rf"F:\Research\AMFEdge\Edge_TreeCover\treeCover_Amazon_2023_dist{frac}.csv"
    outdf.to_csv(outpath, index=False)