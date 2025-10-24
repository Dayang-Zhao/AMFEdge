import sys
sys.path.append(r"D:\ProgramData\VistualStudioCode\AMFEdge")

import pandas as pd
import GlobVars as gv

def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

csv_path = r"F:\Research\AMFEdge\Edge\V2_1sigma\anoVI_Amazon_Edge_Effect_2023.csv"
df = pd.read_csv(csv_path)
df['nirv_scale'] = df['nirv_scale']/1000
# df = df[df['nirv_scale'] <= 6]
ne_df = df[df['Id'].isin(gv.NEGRID_IDS)].copy()
sw_df = df[df['Id'].isin(gv.SWGRID_IDS)].copy()
ne_df = remove_outliers_iqr(ne_df, ['nirv_magnitude', 'nirv_scale'])
sw_df = remove_outliers_iqr(sw_df, ['nirv_magnitude', 'nirv_scale'])

print(ne_df[['nirv_magnitude', 'nirv_scale']].describe())
print(sw_df[['nirv_magnitude', 'nirv_scale']].describe())