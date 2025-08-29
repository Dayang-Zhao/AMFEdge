import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce

offsets = [0, 1, 2, 3, 4, 5]

prefpath1 = r"F:\Research\AMFEdge\EdgeOnset\Amazon_UndistEdge_Effect_0M_2023.csv"
df1_list = []
for offset in offsets:
    df = pd.read_csv(prefpath1.replace('0M', str(offset) + 'M'))
    df['dOnset'] = offset + 1
    df1_list.append(df)
df1 = pd.concat(df1_list, axis=0, ignore_index=True)

prefpath3 = r"F:\Research\AMFEdge\EdgeOnset\anoVI_Amazon_UndistEdge_0M_2023.csv"
df3_list = []
for offset in offsets:
    df = pd.read_csv(prefpath3.replace('0M', str(offset) + 'M'))
    df= df[['Id', 'WD_mean', 'CWD_mean', 'anoCWD_mean']].groupby('Id').mean().reset_index()
    df['dOnset'] = offset + 1
    df3_list.append(df)
df3 = pd.concat(df3_list, axis=0, ignore_index=True)

df = pd.merge(df1, df3, on=['Id', 'dOnset'], how='inner')

path2 = r"F:\Research\AMFEdge\TreeCover\frag_Amazon_2023_dist95.csv"
df2 = pd.read_csv(path2)
path4 = r"F:\Research\AMFEdge\Geography\Amazon_Grids_Hand_stat.csv"
df4 = pd.read_csv(path4)
path5 = r"F:\Research\AMFEdge\Soil\Amazon_Grids_SoilFern_stat.csv"
df5 = pd.read_csv(path5)
path6 = r"F:\Research\AMFEdge\Meteo\Amazon_Grids_anoMeteo_stat.csv"
df6 = pd.read_csv(path6)
path7 = r"F:\Research\AMFEdge\TreeHeight\Amazon_GEDI_treeHeight.csv"
df7 = pd.read_csv(path7)
path8 = r"F:\Research\AMFEdge\Meteo\Amazon_2023_droughtPeriod.csv"
df8 = pd.read_csv(path8)
path9 = r"F:\Research\AMFEdge\Soil\Amazon_SoilGrid_SandSocNitrogen.csv"
df9 = pd.read_csv(path9)
path10 = r"F:\Research\AMFEdge\Meteo\Amazon_Grids_DroughtSeverity_stat.csv"
df10 = pd.read_csv(path10)
path11 = r"F:\Research\AMFEdge\EdgeRH\Amazon_UndistEdge_Effect_2023.csv"
df11 = pd.read_csv(path11)[['Id', 'rh98_scale', 'rh98_magnitude']]

df = reduce(lambda left, right: pd.merge(left, right, on='Id', how='inner'), 
            [df, df2, df4, df5, df6, df7, df8, df9, df10, df11])

outpath = r"F:\Research\AMFEdge\Model\Amazon_Attribution_dOnset.csv"
df.to_csv(outpath, index=False)