import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import scipy.sparse

def to_array(self):
    return self.toarray()

scipy.sparse.spmatrix.A = property(to_array)
from pygam import LinearGAM, s, f, te

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import GlobVars as gv
import Data.save_data as sd

xcols = ['length_mean', 'HAND_mean','undistForest_dist', 
        'SCC_mean', 'sand_mean_mean',
        'anoMCWD_mean','surface_net_solar_radiation_sum_mean',
        'vpd_mean', 'total_precipitation_sum_mean'
        ]

path = r"F:\Research\AMFEdge\GAM\Amazon_Attribution.csv"
df = pd.read_csv(path)
df['nirv_scale'] = df['nirv_scale']/1000
ycols = ['nirv_magnitude', 'nirv_scale']
dst_df = df.dropna(axis=0, subset=xcols+ycols)

for ycol in ycols:
    X = dst_df[xcols].values
    y = dst_df[ycol].values

    # Fit a GAM model with smooth splines on each variable
    gam = LinearGAM(s(0) + s(1) + s(2) 
                    + s(3) + s(4) + s(5) + s(6) 
                    + s(7) + s(8)
                    # + te(1, 2)
                    # + te(0,1) + te(0,2) + te(0,3) 
                    # + te(0,4) + te(0,5) + te(0,7)
                    # + te(7,8) + te(8,9) + te(7,9)
                    ).fit(X, y)
    # Predict
    y_pred = gam.predict(X)
    dst_df[ycol + '_pred'] = y_pred

dst_df = dst_df[['Id']+ycols+[ycol+'_pred' for ycol in ycols]]    
outpath = r"F:\Research\AMFEdge\GAM\gam_prediction.xlsx"
sd.save_pd_as_excel(dst_df, outpath, sheet_name=ycol, index=False)