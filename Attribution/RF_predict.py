import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib.pyplot as plt
plt.ion()

import GlobVars as gv
import Data.save_data as sd
import Attribution.LcoRF as lcorf

SCENARIO = 'SSP5_85'
ROOT_DIR = rf"F:\Research\AMFEdge\CMIP6\GEE\{SCENARIO}"
PRE_FNAME = "anoMeteo_CMIP6_"
BG_DF = pd.read_csv(r"F:\Research\AMFEdge\Model\Amazon_Attribution_v3.csv")
dy_xcols = ['MCWD_mean', 'surface_solar_radiation_downwards_sum_mean',
        'vpd_mean', 'total_precipitation_sum_mean', 'temperature_2m_mean']
static_xcols = ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
        'SCC_mean', 'sand_mean_mean', 'nitrogen_mean_mean']
ycol = 'nirv_magnitude'

def preprocess_data(year):
    dst_path = os.path.join(ROOT_DIR, f"{PRE_FNAME}{year}.csv")
    df = pd.read_csv(dst_path)
    # df['MCWD_mean'] = (df['MCWD_mean'] - 3.18922922998238)/2504.45609867467/1000 * 30 * 24 * 3600
    outdf = df.merge(BG_DF[['Id'] + static_xcols], on='Id')

    # Drop nan.
    outdf.replace(-9999, np.nan, inplace=True)
    outdf = outdf.dropna(subset=dy_xcols+static_xcols)

    return outdf

def main(year, model):
    X_df = preprocess_data(year)
    X = X_df[static_xcols + dy_xcols]
    y = model.predict(X)

    outdf = X_df.copy()[['Id'] + static_xcols + dy_xcols + ['WD_mean', 'total_evaporation_sum_mean']]
    outdf[ycol] = y
    outdf['year'] = year

    return outdf

if __name__ == "__main__":
    df = BG_DF.dropna(subset=['Id'] + dy_xcols + static_xcols + [ycol])
    # df['nirv_magnitude'] = df['nirv_magnitude']*-1
    # df['MCWD_mean'] = df['MCWD_mean']
    df = df[df['nirv_scale'] <= 6000]
    X = df[static_xcols + dy_xcols]
    y = df[ycol]

    model = lcorf.LcoRF()
    # model = xgb.XGBRegressor(objective='reg:squarederror', 
    #         booster='gbtree', random_state=42, n_estimators=40,
    #         max_depth=3, learning_rate=0.2,)
    model.fit(X, y)
    r2 = model.score(X, y)
    print(f"R^2: {r2}")

    years = np.arange(2030, 2100)
    outdf = pd.concat([main(year, model) for year in years], ignore_index=True)

    outpath = rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_pred_{SCENARIO}_v3.csv"
    outdf.to_csv(outpath, index=False)