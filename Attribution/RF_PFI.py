import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle
import json

import matplotlib.pyplot as plt
plt.ion()
import GlobVars as gv

import Data.save_data as sd
import Attribution.LcoRF as lcorf

xcols = ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
         'SCC_mean', 'sand_mean_mean',
        'MCWD_mean', 'surface_solar_radiation_downwards_sum_mean',
         'vpd_mean', 'total_precipitation_sum_mean','temperature_2m_mean']
ycol = 'nirv_scale'

path = r"F:\Research\AMFEdge\Model\Amazon_Attribution.csv"
raw_df = pd.read_csv(path)
df = raw_df.dropna(subset=xcols+[ycol])
df = df[df['nirv_scale'] <= 6000]

X = df[xcols]
y = df[ycol]

# # Standardize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# ----------------- XGB BayersSearchCV. --------------------------------
def tune(outpath):
    regress_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                    booster='gbtree', random_state=42)
    fit_params = {
        'early_stopping_rounds': 10,
        'eval_set':[(X, y)],
        'verbose': True
    }

    search_space = {
        'max_depth': Integer(2, 5, 'uniform'),
        'n_estimators': Integer(30, 60, 'uniform'),
        'learning_rate': Real(0.1, 1.0, 'log-uniform'),
        'scale_pos_weight': Real(1e-6, 1, 'log-uniform'),
        'min_child_weight': Real(0, 10, 'uniform'),
        'max_delta_step': Integer(0, 20, 'uniform'),
        'subsample': Real(0.1, 1.0, 'uniform'),
        'colsample_bytree': Real(0.1, 1.0, 'uniform'),
        'colsample_bylevel': Real(0.1, 1.0, 'uniform'),
        'gamma': Real(1e-9, 0.5, 'log-uniform'),
        'reg_alpha': Real(0, 1.0, 'uniform'),
        'reg_lambda': Real(0, 10, 'uniform')
    }

    bayes_cv_tuner = BayesSearchCV(
        estimator=regress_model,
        search_spaces=search_space,
        fit_params=fit_params,
        cv=5,
        random_state=42,
        n_iter=10,
        verbose=1,
        refit=True,
        n_jobs=-1
    )

    result = bayes_cv_tuner.fit(X, y)
    print(result.best_estimator_)
    print(np.round(result.best_score_, 4))
    print(result.best_params_)

    # Save hyperparameters.
    with open(outpath, "w") as f:
        json.dump(result.best_params_, f)

# outpath = r"F:\Research\AMFEdge\Model\Mnirv_RF_hyperparams.json"
# tune(outpath)
# with open(outpath, "r") as f:
#     best_params = json.load(f)
# model = xgb.XGBRegressor(**best_params)

# ----------------- Train and fit -------------------------------
model = lcorf.LcoRF()
# model = LinearRegression()
model.fit(X, y)
print(f"R^2: {model.score(X, y)}")

# ---------------- permutation importance --------------
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np

r = permutation_importance(model, X, y,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{xcols[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

# Save as CSV
importance_df = pd.DataFrame({
    'Feature': xcols,
    'Importance (ΔR²)': r.importances_mean,
    'Importance std': r.importances_std,
})

outpath = r"F:\Research\AMFEdge\Model\RF_PFI_importance.xlsx"
sd.save_pd_as_excel(importance_df, outpath, sheet_name=ycol, index=False)
