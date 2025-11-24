import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import json

import matplotlib.pyplot as plt
plt.ion()

import GlobVars as gv
import Data.save_data as sd
import Attribution.LcoRF as lcorf

ROOT_DIR = rf"F:\Research\AMFEdge\Comparison"
PRE_FNAME = "Amazon_Grids_"
BG_DF = pd.read_csv(r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution.csv")
dy_xcols = ['MCWD_mean', 'surface_solar_radiation_downwards_sum_mean',
        'vpd_mean', 'total_precipitation_sum_mean', 'temperature_2m_mean']
static_xcols = ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
        'SCC_mean', 'sand_mean_mean', 'histMCWD_mean'] # 'nitrogen_mean_mean'
ycol = 'nirv_magnitude'

def preprocess_data(years:list):
    dfs = []
    for year in years:
        dst_path = os.path.join(ROOT_DIR, f"{PRE_FNAME}{year}_Meteo_stat.csv")
        df = pd.read_csv(dst_path).copy()
        df['year'] = year
        dfs.append(df)

    outdf = pd.concat(dfs, ignore_index=True)

    outdf = outdf.merge(BG_DF[['Id'] + static_xcols], on='Id')

    # Drop nan.
    outdf = outdf.dropna(subset=dy_xcols+static_xcols)

    return outdf

def main(model, years:list):
    X_df = preprocess_data(years)
    X = X_df[static_xcols + dy_xcols]
        
    y = model.predict(X)
    outdf = X_df.copy()[['Id', 'year'] + static_xcols + dy_xcols]
    outdf[ycol] = y

    return outdf

if __name__ == "__main__":
    df = BG_DF.dropna(subset=['Id'] + dy_xcols + static_xcols + [ycol])
    df = df[df['nirv_scale'] <= 6000]
    X = df[static_xcols + dy_xcols]
    y = df[ycol]

    years = [2023]
    model = lcorf.LcoRF()
    model.fit(X, y)
    r2 = model.score(X, y)
    print(f"R2: {r2:.3f}")

    # outdf = main(model, years=years)
    # outdf = outdf.merge(BG_DF, on='Id', suffixes=('_pred', '_true'), how='outer')
    # outdf = outdf.dropna(subset=[ycol+'_pred', ycol+'_true'])

    # Predict raw data.
    raw_df = BG_DF.dropna(subset=['Id'] + dy_xcols + static_xcols).copy()
    X = raw_df[static_xcols + dy_xcols]
    y = model.predict(X)
    outdf = raw_df.copy()[['Id', ycol]]
    outdf[ycol+'_pred'] = y

    outpath = r"F:\Research\AMFEdge\Model\Mnirv_pred_2023.csv"
    outdf.to_csv(outpath, index=False)