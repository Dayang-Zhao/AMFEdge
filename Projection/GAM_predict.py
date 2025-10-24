import sys

sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.sparse

def to_array(self):
    return self.toarray()

scipy.sparse.spmatrix.A = property(to_array)
from pygam import LinearGAM, s, f, te
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from functools import reduce
import operator

import matplotlib.pyplot as plt
plt.ion()

import GlobVars as gv
import Data.save_data as sd
import Attribution.LcoRF as lcorf

SCENARIO = 'SSP2_45'
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

def preprocess_data(model=None, scaler=None):
    dst_path = os.path.join(ROOT_DIR, f"{PRE_FNAME}{model}.csv")
    df = pd.read_csv(dst_path).copy()
    df.replace(-9999, np.nan, inplace=True)

    outdf = df.merge(BG_DF[['Id'] + static_xcols], on='Id')
    outdf = outdf.dropna(subset=dy_xcols+static_xcols)

    # Scale features.
    if scaler is not None:
        X = outdf[static_xcols + dy_xcols]
        X_scaled = scaler.transform(X)
        outdf[static_xcols + dy_xcols] = X_scaled

    return outdf

def main(regressor, scaler):
    outdfs = []

    for model in models:

        X_df = preprocess_data(model, scaler)
        X = X_df[static_xcols + dy_xcols]
            
        y = regressor.predict(X)

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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(X_scaled)

    terms_list = [s(i, n_splines=10) for i in range(X.shape[1])]
    gam_terms = reduce(operator.add, terms_list)
    model = LinearGAM(gam_terms)
    # model = model.gridsearch(X_scaled, y, lam=np.logspace(-3, 3, 10))
    model.fit(X_scaled, y)
    r2 = model.score(X_scaled, y)
    print(f"R^2: {r2}")

    years = np.arange(2015, 2100)
    outdf = main(model, scaler)

    outpath = rf"F:\Research\AMFEdge\CMIP6\Predict\QDM\Mnirv_Edge_GAM_pred_{SCENARIO}.csv"
    outdf.to_csv(outpath, index=False)