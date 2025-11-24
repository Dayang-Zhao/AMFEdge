import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

R2_THRESHOLD = 0.8
DIST_MAX = 6000

# Exponential Function
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Function to Fit Model for Each Group
def fit(xdata, ydata):
    if len(ydata) == 0:
        return np.array([False])
    c_bounds = (ydata.min()-2, ydata.max()+2)
    a_bounds = (ydata[0]-ydata.max()-2, ydata[0]-ydata.min()+2)
    try:
        popt, _ = curve_fit(func, xdata, ydata, maxfev = 10000, 
                # bounds=([-30, -0.02, -5], [30, 0.02,5]),
                bounds=([a_bounds[0], 0, c_bounds[0]], [a_bounds[1], 0.01,c_bounds[1]]),
                check_finite=False, nan_policy='omit')
        return popt
    except RuntimeError:
        return np.array([False])

# Calculate fitting performance
def cal_fit_performance(xdata, ydata, popt):
    y_pred = func(xdata, *popt)
    r2 = r2_score(ydata, y_pred)
    rmse = np.sqrt(mean_squared_error(ydata, y_pred))

    return r2, rmse

# Calculate scale of edge effect, which was the distance where value == intact_value.
def cal_scale(popt):
    x = np.arange(0, DIST_MAX+1, 1)
    y = func(x, *popt)
    intact_value = 0.1*popt[0] + popt[2]
    diff = y - intact_value
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        scale = np.nan
    else:
        scale = x[sign_change[0]]
    return scale

# Calculate magnitude of edge effect, which was the difference between dist=0 and dist=scale.
def cal_magnitude(scale, popt):
    # # Magnitude is the difference between dist=0 and dist=scale
    # x = np.array([0, scale])
    # y = func(x, *popt)
    # magnitude = y[0] - y[1]

    # Magnitude is the difference between dist=0 and the synoptic value.
    magnitude = popt[0]*-1
    return magnitude

def cal_edge_effect(group, x:str, y:str):
    dst_group = group.loc[group['Dist'] > 0]
    # dst_group = group.loc[(group['Dist'] > 0)&(group[y[:-5]+'_skew'] <= 1)&(group[y[:-5]+'_skew'] >= -1)]
    xdata, ydata = dst_group[x].values[1:], dst_group[y].values[1:]
    # intact_value = group[group['Dist'] == -1][y].values[0]

    # Remove NaN values
    mask = ~np.isnan(xdata) & ~np.isnan(ydata)
    xdata, ydata = xdata[mask], ydata[mask]

    # Fit the model
    popt = fit(xdata, ydata)

    if popt.any() == False:
        return pd.Series({'para1': np.nan, 'para2': np.nan, 'para3': np.nan, 
                          'r2': np.nan, 'rmse': np.nan, 'scale': np.nan, 'magnitude': np.nan})
    
    # Calculate fitting performance
    r2, rmse = cal_fit_performance(xdata, ydata, popt)

    # Calculate scale and magnitude of edge effect
    # Case 1: Bad fit
    if (r2 < R2_THRESHOLD):
        return pd.Series({'para1': popt[0], 'para2': popt[1], 'para3': popt[2],
            'r2': r2, 'rmse': rmse, 'scale': np.nan, 'magnitude': np.nan}) 

    scale = cal_scale(popt)

    # Case 2: Good fit but found no scale, in this case, the scale is typically larger than DIST_MAX.
    if np.isnan(scale):
        magnitude = cal_magnitude(DIST_MAX, popt)
        return pd.Series({'para1': popt[0], 'para2': popt[1], 'para3': popt[2],
            'r2': r2, 'rmse': rmse, 'scale': DIST_MAX+1, 'magnitude': magnitude})
    else:
    # Case 3: Good fit and found scale.
        magnitude = cal_magnitude(scale, popt)
        return pd.Series({'para1': popt[0], 'para2': popt[1], 'para3': popt[2],
            'r2': r2, 'rmse': rmse, 'scale': scale, 'magnitude': magnitude}) 

def main(df, ids):
    # Create output dataframe.
    outdf = pd.DataFrame(columns=[
    'ID', 
    'rh98_para1', 'rh98_para2', 'rh98_para3', 'rh98_r2', 'rh98_rmse','rh98_scale', 'rh98_magnitude',
    'rh50_para1', 'rh50_para2', 'rh50_para3', 'rh50_r2', 'rh50_rmse','rh50_scale', 'rh50_magnitude',
    ])
    # outdf = pd.DataFrame(columns=[
    # 'ID', 
    # 'agb_para1', 'agb_para2', 'agb_para3', 'agb_r2', 'agb_rmse','agb_scale', 'agb_magnitude',
    # ])

    # Group by ID and fit model
    def _rename_dict(d:dict, prefix:str):
        return {prefix+'_'+key: value for key, value in d.items()}
    
    for id in ids:
        group = df[df['Id'] == id]
        rh98_out = cal_edge_effect(group[group['rh98_count']>=600], 'Dist', 'rh98_mean')
        rh50_out = cal_edge_effect(group[group['rh50_count']>=600], 'Dist', 'rh50_mean')

        outrow = {'ID':id} | _rename_dict(rh98_out, 'rh98') | _rename_dict(rh50_out, 'rh50')
        # agb_out = cal_edge_effect(group[group['agb_count']>=600], 'Dist', 'agb_mean')
        # outrow = {'ID':id} | _rename_dict(agb_out, 'agb')
        outdf.loc[len(outdf)] = outrow

    return outdf

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_2023_1deg.csv"
    df = pd.read_csv(path)
    dst_df = df.loc[df['Dist']<=DIST_MAX]
    dst_ids = df['Id'].unique()
    # dst_ids = [166, 167]

    outdf = main(dst_df, ids=dst_ids)
    outpath = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_Effect_2023_1deg.csv"
    outdf.rename(columns={'ID':'Id'}).to_csv(outpath, index=False)
