import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce
import shap

from sklearn.inspection import permutation_importance
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

xcols = ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
         'SCC_mean', 'sand_mean_mean',  
        'MCWD_mean', 'surface_solar_radiation_downwards_sum_mean',
         'vpd_mean', 'total_precipitation_sum_mean','temperature_2m_mean']
ycol = 'nirv_scale'

path = r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution_V2.csv"
raw_df = pd.read_csv(path)
df = raw_df.dropna(subset=xcols+[ycol])
df = df[df['nirv_scale'] <= 6000]

#  Split dataset.
X = df[xcols].values
y = df[ycol].values.ravel()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

terms_list = [s(i, n_splines=10) for i in range(X_train.shape[1])]
# terms_list.append(te(0,5))
gam_terms = reduce(operator.add, terms_list)
model = LinearGAM(gam_terms)
model.fit(X_scaled, y)
# 自动搜索最佳 λ（平滑度参数）
# model = model.gridsearch(X_scaled, y, lam=np.logspace(-3, 3, 10))
print(model.summary())
# model.fit(X_scaled, y)

# ---------------- permutation importance --------------
from sklearn.metrics import r2_score
import numpy as np

baseline_score = r2_score(y, model.predict(X_scaled))
print(f"Baseline R²: {baseline_score:.3f}")

# importances = []
# for i in range(X.shape[1]):
#     X_permuted = X.copy()
#     X_permuted[:, i] = np.random.permutation(X[:, i])
#     permuted_score = r2_score(y, model.predict(X_permuted))
#     importances.append(baseline_score - permuted_score)

r = permutation_importance(model, X_scaled, y, scoring='r2',
                           n_repeats=30, random_state=0)

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
    'p-value': model.statistics_['p_values'][:len(xcols)],
})
print(importance_df)

# ---------- SHAP values ----------

explainer = shap.Explainer(model.predict, X_scaled)
shap_values = explainer(X_scaled)
shap.summary_plot(shap_values, X_scaled, plot_type="bar")

print("SHAP values calculated.")

# ----------- Save SHAP values ----------
jshap_df = pd.DataFrame(np.abs(shap_values), columns=X.columns)
shap_importance = shap_df.mean().sort_values(ascending=False)
shap_importance_df = shap_importance.reset_index()
shap_importance_df.columns = ['Feature', 'Mean_Abs_SHAP']

outpath = r"F:\Research\AMFEdge\GAM\gam_shap_importance.xlsx"
sd.save_pd_as_excel(shap_importance_df, outpath, sheet_name=ycol, index=False)
