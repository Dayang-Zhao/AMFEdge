import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import json

import matplotlib.pyplot as plt
plt.ion()

import GlobVars as gv
import Data.save_data as sd
import Attribution.LcoRF as lcorf

SCENARIO = 'SSP5_85'
ROOT_DIR = rf"F:\Research\AMFEdge\CMIP6\GEE\QDM\{SCENARIO}"
PRE_FNAME = "anoMeteo_"
BG_DF = pd.read_csv(r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution.csv")
models = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
            'noresm2_mm', 'miroc6', 'taiesm1',
            'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']
dy_xcols = ['MCWD_mean', 'surface_solar_radiation_downwards_sum_mean',
        'vpd_mean', 'total_precipitation_sum_mean', 'temperature_2m_mean']
static_xcols = ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
        'SCC_mean', 'sand_mean_mean', 'histMCWD_mean'] # 'nitrogen_mean_mean'
ycol = 'nirv_magnitude'

def preprocess_data(model=None):
    dst_path = os.path.join(ROOT_DIR, f"{PRE_FNAME}{model}.csv")
    df = pd.read_csv(dst_path).copy()
    df.replace(-9999, np.nan, inplace=True)

    outdf = df.merge(BG_DF[['Id'] + static_xcols], on='Id')

    # Drop nan.
    outdf = outdf.dropna(subset=dy_xcols+static_xcols)

    return outdf

def main(RF):
    outdfs = []

    for model in models:

        X_df = preprocess_data(model)
        X = X_df[static_xcols + dy_xcols]
            
        y = RF.predict(X)

        outdf = X_df.copy()[['Id', 'year'] + static_xcols + dy_xcols + ['WD_mean', 'total_evaporation_sum_mean']]
        outdf[ycol] = y
        outdf['model'] = model
        outdfs.append(outdf)

    outdf = pd.concat(outdfs, ignore_index=True)

    return outdf

if __name__ == "__main__":
    df = BG_DF.dropna(subset=['Id'] + dy_xcols + static_xcols + [ycol])
    df = df[df['nirv_scale'] <= 6000]
    X = df[static_xcols + dy_xcols]
    y = df[ycol]

    model = lcorf.LcoRF()
    # outpath = r"F:\Research\AMFEdge\Model\Mnirv_RF_hyperparams.json"
    # with open(outpath, "r") as f:
    #     best_params = json.load(f)
    # model = xgb.XGBRegressor(**best_params)
    model.fit(X, y)
    r2 = model.score(X, y)
    print(f"R2: {r2:.3f}")

    years = np.arange(2015, 2100)
    outdf = main(model)

    outpath = rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_Edge_pred_{SCENARIO}.csv"
    outdf.to_csv(outpath, index=False)