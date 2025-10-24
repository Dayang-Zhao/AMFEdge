import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# mod_rootdir = r"F:\Research\AMFEdge\EdgeMod"
# years = [2005, 2010, 2015, 2023]
# dfs = []
# for year in years:
#     path = rf"{mod_rootdir}\anoVI_Amazon_Edge_{year}_diff.csv"
#     df = pd.read_csv(path)
#     df['year'] = year
#     dfs.append(df)
# mod_df = pd.concat(dfs, ignore_index=True)

# outpath = r"F:\Research\AMFEdge\Comparison\NIRv_MODIS_Edge_Diff.csv"
# mod_df.to_csv(outpath, index=False)

mod_path = r"F:\Research\AMFEdge\Comparison\NIRv_MODIS_Edge_Diff.csv"
mod_df = pd.read_csv(mod_path)
pred_path = r"F:\Research\AMFEdge\Comparison\MNIRv_Predictions.csv"
pred_df = pd.read_csv(pred_path)
merged_df = mod_df.merge(pred_df, on=['Id', 'year'], suffixes=('_modis', '_pred'))
merged_df = merged_df.dropna(subset=['dNIRv_10_50', 'nirv_magnitude'])
print(merged_df)