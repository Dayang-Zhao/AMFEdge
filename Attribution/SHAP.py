import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from  sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import Data.save_data as sd

xcols = ['HAND_mean', 
         'rh98_scale', 'rh98_magnitude', 
         'SCC_mean', 'sand_mean_mean', 'nitrogen_mean_mean',
        'MCWD_mean',
        #  'length_mean',
        #  'WD_mean','histMCWD_mean',
         'surface_net_solar_radiation_sum_mean',
         'vpd_mean', 'total_precipitation_sum_mean',
        'temperature_2m_mean', 
         ]
ycol = 'nirv_scale'

path = r"F:\Research\AMFEdge\Model\Amazon_Attribution.csv"
raw_df = pd.read_csv(path)
df = raw_df.dropna(subset=xcols+[ycol])
df['nirv_scale'] = df['nirv_scale']/1000
df = df[df['nirv_scale'] <= 6]
X = df[xcols]
y = df[ycol]
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
# X_std = scaler_X.fit_transform(X)
# y_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ---------- Split dataset and train model ----------
# X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, random_state=42)
model = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=2, random_state=42)
# model = LinearRegression()
model.fit(X, y)

# ---------- Test the model ----------
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R^2: {r2:.4f}")
mse = mean_squared_error(y, y_pred)

# ---------- SHAP values ----------
explainer = shap.TreeExplainer(model, X)
shap_values = explainer.shap_values(X)

# ----------- SHAP summary plot ----------
shap.summary_plot(shap_values, X, plot_type="bar")

# ----------- Save SHAP values ----------
shap_df = pd.DataFrame(np.abs(shap_values), columns=X.columns)
shap_importance = shap_df.mean().sort_values(ascending=False)
shap_importance_df = shap_importance.reset_index()
shap_importance_df.columns = ['Feature', 'Mean_Abs_SHAP']

outpath = "F:\Research\AMFEdge\Model\RF_shap_importance.xlsx"
sd.save_pd_as_excel(shap_importance_df, outpath, sheet_name=ycol, index=False)

