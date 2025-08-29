import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce

def merge_df(year):
    path1 = rf"F:\Research\AMFEdge\EdgeMod\anoVI_Amazon_UndistEdge_{year}_diff.csv"
    df1 = pd.read_csv(path1)
    path2 = r"F:\Research\AMFEdge\TreeCover\frag_Amazon_2023_dist95.csv"
    df2 = pd.read_csv(path2)
    path3 = rf"F:\Research\AMFEdge\EdgeMod\anoVI_Amazon_UndistEdge_{year}.csv"
    df3 = pd.read_csv(path3)
    df3= df3[['Id', 'MCWD_mean', 'anoMCWD_mean']].groupby('Id').mean().reset_index()
    path4 = r"F:\Research\AMFEdge\Geography\Amazon_Grids_Hand_stat.csv"
    df4 = pd.read_csv(path4)
    path5 = r"F:\Research\AMFEdge\Soil\Amazon_Grids_SoilFern_stat.csv"
    df5 = pd.read_csv(path5)
    path6 = rf"F:\Research\AMFEdge\Meteo\Amazon_Grids_{year}_anoMeteo_stat.csv"
    df6 = pd.read_csv(path6)
    path7 = r"F:\Research\AMFEdge\TreeHeight\Amazon_GEDI_treeHeight.csv"
    df7 = pd.read_csv(path7)
    path9 = r"F:\Research\AMFEdge\Soil\Amazon_SoilGrid_SandSocNitrogen.csv"
    df9 = pd.read_csv(path9)
    path11 = r"F:\Research\AMFEdge\EdgeRH\Amazon_UndistEdge_Effect_2023.csv"
    df11 = pd.read_csv(path11)[['Id', 'rh98_scale', 'rh98_magnitude']]

    df = reduce(lambda left, right: pd.merge(left, right, on='Id', how='inner'),
                [df1, df2, df3, df4, df5, df6, df7, df9, df11])
    df['year'] = year

    return df

if __name__ == '__main__':
    years = [2005, 2010, 2015, 2023]
    dfs = []
    for year in years:
        df = merge_df(year)
        dfs.append(df)
    
    outdf = pd.concat(dfs, ignore_index=True)
    outdf.to_csv(r"F:\Research\AMFEdge\Model\Amazon_Mod_Attribution.csv", index=False)