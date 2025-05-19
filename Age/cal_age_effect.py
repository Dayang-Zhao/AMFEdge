import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

R2_THRESHOLD = 0.5
AGE_MAX = 34

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
                bounds=([a_bounds[0], 0, c_bounds[0]], [a_bounds[1], 10,c_bounds[1]]),
                check_finite=False, nan_policy='omit')
        return popt
    except RuntimeError:
        return False

# Calculate fitting performance
def cal_fit_performance(xdata, ydata, popt):
    y_pred = func(xdata, *popt)
    r2 = r2_score(ydata, y_pred)
    rmse = np.sqrt(mean_squared_error(ydata, y_pred))

    return r2, rmse

# Calculate scale of edge effect, which was the ageance where value == intact_value.
def cal_scale(popt):
    x = np.arange(0, AGE_MAX+1, 1)
    y = func(x, *popt)
    intact_value = 0.1*popt[0] + popt[2]
    diff = y - intact_value
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        scale = np.nan
    else:
        scale = x[sign_change[0]]
    return scale

# Calculate magnitude of edge effect, which was the difference between age=0 and age=scale.
def cal_magnitude(scale, popt):
    x = np.array([0, scale])
    y = func(x, *popt)
    magnitude = y[0] - y[1]
    return magnitude

def cal_edge_effect(group, x:str, y:str):
    dst_group = group.loc[group['age'] > 0]
    # dst_group = group.loc[(group['age'] > 0)&(group[y[:-5]+'_skew'] <= 1)&(group[y[:-5]+'_skew'] >= -1)]
    xdata, ydata = dst_group[x].values, dst_group[y].values
    # intact_value = group[group['age'] == -1][y].values[0]

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

    # Case 2: Good fit but found no scale, in this case, the scale is typically larger than AGE_MAX.
    if np.isnan(scale):
        magnitude = cal_magnitude(AGE_MAX, popt)
        return pd.Series({'para1': popt[0], 'para2': popt[1], 'para3': popt[2],
            'r2': r2, 'rmse': rmse, 'scale': AGE_MAX+1, 'magnitude': magnitude})
    else:
    # Case 3: Good fit and found scale.
        magnitude = cal_magnitude(scale, popt)
        return pd.Series({'para1': popt[0], 'para2': popt[1], 'para3': popt[2],
            'r2': r2, 'rmse': rmse, 'scale': scale, 'magnitude': magnitude}) 

def main(dfs, edge_types):

    # Group by ID and fit model
    def _rename_dict(d:dict, prefix:str, suffix:str):
        return {prefix+'_'+key+'_'+suffix: value for key, value in d.items()}
    
    outrows = []
    for i, df in enumerate(dfs):
        edge_type = edge_types[i]
        for grid in ['pgrid', 'ngrid']:
            nirv_out = cal_edge_effect(df, 'age', 'fNIRv_mean_'+grid)
            evi_out = cal_edge_effect(df, 'age', 'fEVI_mean_'+grid)
            ndwi_out = cal_edge_effect(df, 'age', 'fNDWI_mean_'+grid)

            outrow = {'edge_type':edge_type} | _rename_dict(nirv_out, 'nirv', grid)\
                | _rename_dict(evi_out, 'evi', grid) | _rename_dict(ndwi_out, 'ndwi', grid)
            outrows.append(pd.DataFrame(outrow, index=[0]))

    outdf = pd.concat(outrows, axis=0, ignore_index=True)
    return outdf

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\EdgeAge\anoVI_panAmazon_dspecUndistEdge_2023_age.xlsx"
    edge_types = ['grass', 'crop', 'water', 'nonveg']
    dfs = [pd.read_excel(path, sheet_name=edge_type) for edge_type in edge_types]

    outdf = main(dfs, edge_types=edge_types)
    outpath = r"F:\Research\AMFEdge\EdgeAge\anoVI_panAmazon_dspecUndistEdge_2023_age_fitting.xlsx"
    outdf.to_csv(outpath, index=False)
