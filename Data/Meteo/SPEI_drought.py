import numpy as np
import pandas as pd

import osgeo
import xarray as xr

import matplotlib.pyplot as plt 

def preprocess(ds:xr.Dataset):


path = r"F:\Research\AMFEdge\Meteo\Meta\spei01.nc"
ds = xr.open_dataset(path)
spei = ds['spei']

# 设置干旱阈值（通常 -1）
threshold = -1.0

# 判断干旱状态
is_drought = spei < threshold

# 找到干旱开始与结束
# 当状态从 False → True 时，是开始
# 当状态从 True → False 时，是结束
drought_start = (is_drought.astype(int).diff("time") == 1)
drought_end = (is_drought.astype(int).diff("time") == -1)

# 提取时间点
start_times = spei.time.where(drought_start, drop=True)
end_times = spei.time.where(drought_end, drop=True)

# 若干旱未结束（即最后仍为干旱），可补齐
if len(end_times) < len(start_times):
    end_times = end_times.append(xr.DataArray([spei.time[-1].values], dims="time"))

# 计算持续时间与强度
events = []
for s, e in zip(start_times.values, end_times.values):
    subset = spei.sel(time=slice(s, e))
    duration = len(subset)
    intensity = -subset.where(subset < threshold).sum().item()
    events.append({"start": pd.Timestamp(s), "end": pd.Timestamp(e),
                   "duration_months": duration, "intensity": intensity})

drought_events = pd.DataFrame(events)
print(drought_events)
