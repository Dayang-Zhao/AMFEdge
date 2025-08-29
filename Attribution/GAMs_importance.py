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

import matplotlib.pyplot as plt
plt.ion()
import GlobVars as gv

import Data.save_data as sd

xcols = ['length_mean', 'HAND_mean','undistForest_dist', 
         'SCC_mean', 'sand_mean_mean',
         'anoMCWD_mean','surface_net_solar_radiation_sum_mean',
         'vpd_mean', 'total_precipitation_sum_mean'
         ]
ycol = 'ndwi_scale'

path = r"F:\Research\AMFEdge\GAM\Amazon_Attribution.csv"
df = pd.read_csv(path)
df = df.dropna(subset=xcols+[ycol])
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
# X = scaler_X.fit_transform(df[xcols])
# y = scaler_y.fit_transform(df[ycol].values.reshape(-1, 1)).ravel()
X = df[xcols].values
y = df[ycol].values

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

# baseline_score = r2_score(y, gam.predict(X))

# importances = []
# for i in range(X.shape[1]):
#     X_permuted = X.copy()
#     X_permuted[:, i] = np.random.permutation(X[:, i])
#     permuted_score = r2_score(y, gam.predict(X_permuted))
#     importances.append(baseline_score - permuted_score)

r = permutation_importance(gam, X, y,
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
    'p-value': gam.statistics_['p_values'][:len(xcols)],
})

outpath = r"F:\Research\AMFEdge\GAM\gam_importance.xlsx"
sd.save_pd_as_excel(importance_df, outpath, sheet_name=ycol, index=False)
