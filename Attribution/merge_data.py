import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce


path1 = r"F:\Research\AMFEdge\Edge\Undist\anoVI_Amazon_undistEdge_Effect_2023.csv"
df1 = pd.read_csv(path1)
path2 = r"F:\Research\AMFEdge\Meteo\Amazon_Grids_2023_histMCWD_stat.csv"
df2 = pd.read_csv(path2).rename(columns={'MCWD_mean': 'histMCWD_mean'})[['Id', 'histMCWD_mean']]
path3 = r"F:\Research\AMFEdge\Edge\Undist\anoVI_Amazon_undistEdge_2023.csv"
df3 = pd.read_csv(path3)
df3= df3[['Id', 'MCWD_mean', 'anoMCWD_mean']].groupby('Id').mean().reset_index()
path4 = r"F:\Research\AMFEdge\Geography\Amazon_Grids_Hand_stat.csv"
df4 = pd.read_csv(path4)
path5 = r"F:\Research\AMFEdge\Soil\Amazon_Grids_SoilFern_stat.csv"
df5 = pd.read_csv(path5)
path6 = r"F:\Research\AMFEdge\Meteo\Amazon_Grids_2023_anoMeteo_stat.csv"
df6 = pd.read_csv(path6)
path9 = r"F:\Research\AMFEdge\Soil\Amazon_SoilGrid_SandSocNitrogen.csv"
df9 = pd.read_csv(path9)
path11 = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_Effect_2023.csv"
df11 = pd.read_csv(path11)[['Id', 'rh98_scale', 'rh98_magnitude', 'rh50_scale', 'rh50_magnitude']]
path12 = r"F:\Research\AMFEdge\EdgeNum\Area_Amazon_Edge_2023.csv"
df12 = pd.read_csv(path12)

df = reduce(lambda left, right: pd.merge(left, right, on='Id', how='inner'), 
            [df1, df2, df3, df4, df5, df6, df9, df11, df12])

outpath = r"F:\Research\AMFEdge\Model\Amazon_undistEdge_Attribution.csv"
df.to_csv(outpath, index=False)