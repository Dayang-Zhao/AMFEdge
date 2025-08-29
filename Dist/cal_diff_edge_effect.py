import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

path1 = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
df1 = pd.read_csv(path1)[['Id', 'nirv_magnitude', 'nirv_scale']]

path2 = r"F:\Research\AMFEdge\Edge_Onset\Amazon_Onset_UndistEdge_Effect_2023.csv"
df2 = pd.read_csv(path2)[['Id', 'nirv_magnitude', 'nirv_scale']]

merged_df = df1.merge(df2, on="Id", how="inner", suffixes=('_period', '_onset'))
merged_df['nirv_magnitude_diff'] = merged_df['nirv_magnitude_period'] - merged_df['nirv_magnitude_onset']
merged_df['nirv_scale_diff'] = merged_df['nirv_scale_period'] - merged_df['nirv_scale_onset']

merged_df.to_csv(r"F:\Research\AMFEdge\Edge_Onset\Amazon_diff_UndistEdge_Effect_2023.csv", index=False)

print(merged_df)