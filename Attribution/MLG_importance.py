import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import numpy as np
import pandas as pd
from functools import reduce

from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
plt.ion()
import GlobVars as gv

import Data.save_data as sd

xcols = ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
         'SCC_mean', 'sand_mean_mean', 
        'MCWD_mean', 'surface_solar_radiation_downwards_sum_mean',
         'vpd_mean', 'total_precipitation_sum_mean','temperature_2m_mean']
ycol = 'nirv_magnitude'

path = r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution_V2.csv"
raw_df = pd.read_csv(path)
df = raw_df.dropna(subset=xcols+[ycol])
df = df[df['nirv_scale'] <= 6000]

X = df[xcols]
y = df[ycol]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a GAM model with smooth splines on each variable
model = LinearRegression().fit(X_scaled, y)
print("截距 (β0):", model.intercept_)
print("回归系数 (β1, β2, β3):", model.coef_)

# Predict
y_pred = model.predict(X_scaled)

# # Plot partial dependencies
# rows, cols = 2, 4
# fig, axs = plt.subplots(rows, cols)
# for i in range(rows):
#     for j in range(cols):
#         ax = axs[i, j]
#         term_i = i * cols + j
#         XX = gam.generate_X_grid(term=term_i)
#         ax.plot(XX[:, term_i], gam.partial_dependence(term=term_i, X=XX))
#         ax.plot(XX[:, term_i], gam.partial_dependence(term=term_i, X=XX, width=0.95)[1], c='r', ls='--')
#         ax.set_title(f'Effect of {xcols[term_i]}')
# plt.tight_layout()
# plt.show()

# # ---------------- SHAP ------------------------
# import shap

# explainer = shap.Explainer(gam.predict, X[:80])
# shap_values = explainer(X)
# shap.summary_plot(shap_values, X, plot_type="bar")

# print("SHAP values calculated.")

# ---------------- permutation importance --------------
from sklearn.metrics import r2_score
import numpy as np

baseline_score = r2_score(y, model.predict(X_scaled))
print(f"Baseline R²: {baseline_score:.3f}")

# importances = []
# for i in range(X.shape[1]):
#     X_permuted = X.copy()
#     X_permuted[:, i] = np.random.permutation(X[:, i])
#     permuted_score = r2_score(y, gam.predict(X_permuted))
#     importances.append(baseline_score - permuted_score)

r = permutation_importance(model, X_scaled, y,
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
    # 'p-value': model.statistics_['p_values'][:len(xcols)],
})
print(importance_df)

outpath = r"F:\Research\AMFEdge\GAM\gam_importance.xlsx"
sd.save_pd_as_excel(importance_df, outpath, sheet_name=ycol, index=False)
