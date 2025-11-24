import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
import xgboost as xgb
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pickle
import json

import matplotlib.pyplot as plt
plt.ion()
import GlobVars as gv

import Data.save_data as sd
import Attribution.LcoRF as lcorf

xcols = [
    'MCWD_mean', 'histMCWD_mean', 'vpd_mean', 'total_precipitation_sum_mean',
    'temperature_2m_mean', 'surface_solar_radiation_downwards_sum_mean', 
    'rh98_scale', 'rh98_magnitude', 
    'HAND_mean',
    'SCC_mean', 'sand_mean_mean', 
    ]
ycol = 'nirv_magnitude'

path = r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution_1deg.csv"
raw_df = pd.read_csv(path)
df = raw_df.dropna(subset=xcols+[ycol])
df = df[df['nirv_scale'] <= 6000]

X = df[xcols]
y = df[ycol]

# ----------------- XGB BayersSearchCV. --------------------------------
def tune(outpath):
    regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    search_space = {
        'max_depth': Integer(2, 8),
        'n_estimators': Integer(10, 500),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'min_child_weight': Real(1, 10),
        'subsample': Real(0.1, 1.0),
        'colsample_bytree': Real(0.1, 1.0),
        'gamma': Real(0, 0.3),
        'reg_alpha': Real(0, 1.0),
        'reg_lambda': Real(0, 10),
    }
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    bayes_cv_tuner = BayesSearchCV(
        regressor,
        search_spaces=search_space,
        scoring='r2',
        n_iter=25,
        cv=cv.split(X, y_binned),
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    result = bayes_cv_tuner.fit(X, y)
    print("Best estimator:", result.best_estimator_)
    print("Best R²:", np.round(result.best_score_, 4))
    print("Best params:", result.best_params_)

    # 保存最优参数
    with open(outpath, "w") as f:
        json.dump(result.best_params_, f, indent=2)

outpath = r"F:\Research\AMFEdge\Model\Mnirv_RF_hyperparams.json"
tune(outpath)
with open(outpath, "r") as f:
    best_params = json.load(f)
model = xgb.XGBRegressor(**best_params)
# ---------- Bootstrap with stratified sampling -----------------
n_samples = X.shape[0]
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
unique_bins = np.unique(y_binned)

B = 100  # Bootstrap 重复次数
r2_scores = []

for b in range(B):
    train_idx = []

    # 分层抽样：每个 bin 内有放回抽样
    for bin_val in unique_bins:
        bin_idx = np.where(y_binned == bin_val)[0]
        n_bin = len(bin_idx)
        sampled_idx = np.random.choice(bin_idx, size=n_bin, replace=True)
        train_idx.extend(sampled_idx)

    train_idx = np.array(train_idx)
    
    # OOB 样本
    oob_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    if len(oob_idx) == 0:
        continue  # 避免空测试集
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[oob_idx], y.iloc[oob_idx]

    # 训练 XGBoost
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))

# 输出
r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)
r2_ci = np.percentile(r2_scores, [2.5, 97.5])

print(f"Stratified Bootstrap R²: {r2_mean:.3f} ± {r2_std:.3f}")
print(f"95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")

# ----------------- Train and fit -------------------------------
model.fit(X, y)
print(f"R^2: {model.score(X, y)}")

# ---------------- permutation importance --------------
import numpy as np

r = permutation_importance(model, X, y,
                           n_repeats=30,
                           random_state=0, scoring='r2')

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

outpath = r"F:\Research\AMFEdge\Model\XGB_Edge_PFI_importance.xlsx"
sd.save_pd_as_excel(importance_df, outpath, sheet_name=ycol, index=False)
