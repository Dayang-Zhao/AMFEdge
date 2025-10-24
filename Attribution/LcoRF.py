"""Random Forest + Linear correction"""
import pandas as pd

from scipy.stats import linregress
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
plt.ion()

class LcoRF:
    def __init__(self, n_estimators=300, max_depth=3, min_samples_split=2, 
                 min_samples_leaf=2):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self._estimator_type = "regressor"

    def fit(self, X, y):
        self.model.fit(X, y)
        RF_y_pred = self.model.predict(X)
        linear_reg = linregress(RF_y_pred, y)
        self.linear_reg = linear_reg
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")  # 如果没fit会报错
        RF_y_pred = self.model.predict(X)
        y_pred = RF_y_pred * self.linear_reg.slope + self.linear_reg.intercept
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
def grid_search(model, X, y, cv=5):
    param_grid = {
        'n_estimators': range(10, 41),     # 树的数量
        'max_depth': range(2,6),      # 最大深度
        'min_samples_split': range(2,6),     # 最小划分样本数
        'min_samples_leaf': range(1,6),       # 叶子节点最小样本数
    }

    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='r2')
    grid_search.fit(X, y)
  
    return grid_search.best_params_, grid_search.best_score_

if __name__ == '__main__':
    xcols = ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
            'SCC_mean', 'sand_mean_mean',
            'MCWD_mean', 'surface_solar_radiation_downwards_sum_mean',
            'vpd_mean', 'total_precipitation_sum_mean','temperature_2m_mean']
    ycol = 'nirv_magnitude'

    path = r"F:\Research\AMFEdge\Model\Amazon_Attribution.csv"
    raw_df = pd.read_csv(path)
    df = raw_df.dropna(subset=xcols+[ycol])
    df = df[df['nirv_scale'] <= 6000]

    X = df[xcols].values
    y = df[ycol].values

    model = LcoRF()
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f"R^2: {model.linear_reg.rvalue**2:.4f}, \
          Slope: {model.linear_reg.slope:.4f}, \
          Intercept: {model.linear_reg.intercept:.4f}")