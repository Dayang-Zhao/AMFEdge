import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import pandas as pd

import matplotlib.pyplot as plt

import GlobVars as gv

# path = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_Effect_2023.csv"
path = r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution.csv"
df = pd.read_csv(path)
# df['edge_rh98'] = df['rh98_para3'] + df['rh98_para1']

dst_df = df.loc[df['Id'].isin(gv.AZGRID_IDS)]
print(dst_df)