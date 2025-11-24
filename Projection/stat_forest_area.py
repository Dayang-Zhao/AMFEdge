import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"F:\Research\AMFEdge\CMIP6\Predict\diff_2015@2090.csv"
df = pd.read_csv(path)

path2 = r"F:\Research\AMFEdge\EdgeNum\Area_Amazon_Edge_2023.csv"
area_df = pd.read_csv(path2)
area_df['forest_count'] = area_df['edge_count'] + area_df['interior_count']
sum_forest_count = area_df['forest_count'].sum()

df = df.merge(area_df[['Id', 'forest_count']], on='Id', how='left')
dst_df = df[(df['diff_sign_prec_nirv_magnitude']>=0.8)&(df['dnirv_magnitude']>=0)].copy()
sum_df = dst_df.groupby('scenario').sum().reset_index()
sum_df['total_area'] = sum_forest_count
sum_df['prec'] = sum_df['forest_count']*100 / sum_df['total_area']
print(dst_df)