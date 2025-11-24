import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
from sklearn.metrics import r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle
import json

import matplotlib.pyplot as plt
plt.ion()
import GlobVars as gv

import Data.save_data as sd
import Attribution.LcoRF as lcorf

xcols = [
    'vpd_mean', 'total_precipitation_sum_mean',
    'temperature_2m_mean', 'surface_solar_radiation_downwards_sum_mean', 
    'rh98_scale', 'rh98_magnitude', 
    'HAND_mean', 'histMCWD_mean', 
    'SCC_mean', 'sand_mean_mean', 
    ]
groups = ['Drought']*4 + ['Forest structure']*2 + ['Hydrology']*2 + ['Soil']*2
ycol = 'nirv_scale'

path = r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution.csv"
raw_df = pd.read_csv(path)
raw_df['dMCWD_mean'] = raw_df['MCWD_mean'] - raw_df['histMCWD_mean']
df = raw_df.dropna(subset=xcols+[ycol])
df['nirv_scale'] = df['nirv_scale'] / 1000
df = df[df['nirv_scale'] <= 6].reset_index(drop=True)

X = df[xcols].copy()
y = df[ycol].copy()
model = lcorf.LcoRF()
model.fit(X, y)
print("R²:", model.score(X, y))

# ----------------- Cross-validation --------------------------------
# model = lcorf.LcoRF()
# y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(model, X, y, cv=cv.split(X, y_binned), scoring='r2')
# rmse_scores = -cross_val_score(model, X, y, cv=cv.split(X, y_binned), scoring='neg_mean_absolute_error')
# print("Repeated 5x5 CV R2: %.3f ± %.3f" % (scores.mean(), scores.std()))
# print("Repeated 5x5 CV RMSE: %.3f ± %.3f" % (rmse_scores.mean(), rmse_scores.std()))

# ---------------- SHAP ------------------
import shap
explainer = shap.TreeExplainer(model.model)  # inner RF model
shap_values = explainer.shap_values(X)

# Adjust SHAP values by the linear correction
scaled_shap_values = np.array(shap_values) * model.linear_reg.slope
shap_summary = pd.DataFrame({
    'Feature': X.columns,
    'Group': groups,
    'mean_abs_shap': np.abs(scaled_shap_values).mean(axis=0)
})
outpath = r"F:\Research\AMFEdge\Model\RF_Edge_SHAP_outMCWD.xlsx"
sd.save_pd_as_excel(shap_summary, outpath, sheet_name=ycol, index=False)
# shap.summary_plot(scaled_shap_values, X, plot_type="bar")

# # ---------------- permutation importance --------------
# from sklearn.metrics import r2_score, root_mean_squared_error
# import numpy as np

# r = permutation_importance(model, X, y,
#                            n_repeats=30,
#                            random_state=0)

# for i in r.importances_mean.argsort()[::-1]:
#     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#         print(f"{xcols[i]:<8}"
#               f"{r.importances_mean[i]:.3f}"
#               f" +/- {r.importances_std[i]:.3f}")

# # Save as CSV
# importance_df = pd.DataFrame({
#     'Feature': xcols,
#     'Group': groups,
#     'Importance (ΔR²)': r.importances_mean,
#     'Importance std': r.importances_std,
# })
# importance_df['Importance percent'] = importance_df['Importance (ΔR²)'] / np.sum(importance_df['Importance (ΔR²)']) * 100
# print(importance_df)

# outpath = r"F:\Research\AMFEdge\Model\RF_Edge_PFI_importance_outMCWD.xlsx"
# sd.save_pd_as_excel(importance_df, outpath, sheet_name=ycol, index=False)
