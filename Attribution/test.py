import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

path = r"F:\Research\AMFEdge\Model\Amazon_Edge_Attribution.csv"
df = pd.read_csv(path)
xcols, ycol = ['rh98_scale'], 'nirv_scale'
df = df.dropna(subset=xcols+[ycol])
df = df[df['nirv_scale'] <= 6000]
df = df[df['rh98_scale'] <= 6000]

X = df[xcols].values
y = df[ycol].values

# 假设 X, y 是 numpy array，shape (n_samples, n_features)
# Add intercept
X_sm = sm.add_constant(X)   # adds column of ones

# Fit robust linear model with Huber's psi
rlm_model = sm.RLM(y, X_sm, M=sm.robust.norms.HuberT())
rlm_res = rlm_model.fit()   # 可以传 maxiter, tol 等参数

print(rlm_res.summary())    # 这是一个有帮助的概览（但注意 summary 在 RLM 中没有像 OLS 那么全面的 p-value 部分）

# 获取参数与协方差 -> 计算标准误差和近似 p 值
params = rlm_res.params               # array (p,)
cov = rlm_res.cov_params()            # covariance matrix of params (p,p)
bse = np.sqrt(np.diag(cov))           # 标准误
z_scores = params / bse
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))  # 两侧检验

for name, coef, se, z, p in zip(['const'] + xcols, params, bse, z_scores, p_values):
    print(f"{name:10s} coef={coef:.4f} se={se:.4f} z={z:.3f} p={p:.3e}")


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppose you already fit:
# model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
y_pred = rlm_res.predict(X_sm)

# Compute metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print(f"R² = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"MAE = {mae:.3f}")