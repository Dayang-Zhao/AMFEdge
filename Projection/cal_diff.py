from functools import reduce
import pandas as pd
from scipy.stats import linregress

import matplotlib.pyplot as plt

start_period = range(2015, 2025)
end_period = range(2090, 2100)

# -------------- Linear regression to calculate trend and p-value --------------
# def linreg(group):
#     res = linregress(group["year"], group[ycol])
#     return pd.Series({"slope": res.slope, "p": res.pvalue})

# def cal_sign_prec(group):
#     # Caculate the percentage of consistent and significant sign trends among models.
#     pos_sign = ((group['slope'] > 0) & (group['p'] < 0.05)).sum()
#     neg_sign = ((group['slope'] < 0) & (group['p'] < 0.05)).sum()
#     total = len(group['model'].unique())
#     pos_prec = pos_sign / total
#     neg_prec = neg_sign / total
#     prec = max(pos_prec, neg_prec)

#     return pd.Series({"sign_prec": prec})

# def main(df):
#     df = df.dropna(subset=[ycol])
#     result = df.groupby(["Id", "model"]).apply(linreg).reset_index()
#     sign_prec = result.groupby("Id").apply(cal_sign_prec).reset_index()

#     return sign_prec

# -------------- Calculate the difference between end and start period --------------
def remove_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series >= lower) & (series <= upper)]

def cal_diff2(df, dst_var):
    # Remove outliers based on IQR for each Id
    # df_clean = (
    #     df.groupby(['Id', 'model'], group_keys=False)
    #     .apply(lambda g: g.loc[remove_outliers_iqr(g[dst_var]).index])
    # )
    df_start = df[df['year'].isin(start_period)].groupby(['Id', 'model'])[dst_var].mean().reset_index()
    df_end = df[df['year'].isin(end_period)].groupby(['Id', 'model'])[dst_var].mean().reset_index()
    df_merged = pd.merge(df_start, df_end, on=['Id', 'model'], suffixes=('_start', '_end'))
    df_merged['d'+dst_var] = df_merged[dst_var+'_end'] - df_merged[dst_var+'_start']

    return df_merged

def cal_diff1(df, dst_var):
    df_start = df[df['year'].isin(start_period)].groupby(['Id'])[dst_var].mean().reset_index()
    df_end = df[df['year'].isin(end_period)].groupby(['Id'])[dst_var].mean().reset_index()
    df_merged = pd.merge(df_start, df_end, on=['Id'], suffixes=('_start', '_end'))
    df_merged['d'+dst_var] = df_merged[dst_var+'_end'] - df_merged[dst_var+'_start']

    return df_merged

def cal_sign_prec(group, dst_var):
    # Caculate the percentage of consistent and significant sign trends among models.
    pos_sign = (group[dst_var] > 0).sum()
    neg_sign = (group[dst_var] < 0).sum()
    # total = len(group['model'].unique())
    total = 10
    pos_prec = pos_sign / total
    neg_prec = neg_sign / total
    prec = max(pos_prec, neg_prec)

    return pd.Series({"sign_prec": prec})

def main(df, dst_var):
    df = df[['Id', 'model', 'year', dst_var]].dropna(subset=[dst_var])

    # Multi-year average and sign consistency among models.
    mean_df1 = df.groupby(['Id'])[dst_var].mean().reset_index()
    mean_df = df.groupby(['Id', 'model'])[dst_var].mean().reset_index()
    mean_sign_prec = mean_df.groupby("Id").apply(cal_sign_prec, dst_var=dst_var).reset_index()\
        .rename(columns={'sign_prec': 'mean_sign_prec_'+dst_var})
    mean_df1 = mean_df1.merge(mean_sign_prec, on='Id')

    # Difference between end and start period and sign consistency among models.
    diff_df1 = cal_diff1(df, dst_var)
    diff_df = cal_diff2(df, dst_var)
    diff_sign_prec = diff_df.groupby("Id").apply(cal_sign_prec, dst_var='d'+dst_var).reset_index()\
        .rename(columns={'sign_prec': 'diff_sign_prec_'+dst_var})
    diff_df1 = diff_df1.merge(diff_sign_prec, on='Id')

    outdf = mean_df1.merge(diff_df1, on=['Id'])

    return outdf


if __name__ == "__main__":
    dst_vars = ['nirv_magnitude', 'MCWD_mean', 'relAnoMCWD_mean', 'temperature_2m_mean']
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']

    outdfs = []
    for scenario in scenarios:
        df = pd.read_csv(rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_Edge_pred_{scenario}.csv")
        df['MCWD_mean'] = df['MCWD_mean']*100/df['histMCWD_mean']
        var_dfs = []
        for dst_var in dst_vars:
            var_df = main(df, dst_var)
            var_dfs.append(var_df)
        scenario_df = reduce(lambda left, right: pd.merge(left, right, on=['Id'], how='inner'), var_dfs)
        scenario_df['scenario'] = scenario
        outdfs.append(scenario_df)
    outdf = pd.concat(outdfs, ignore_index=True)

    outpath = r"F:\Research\AMFEdge\CMIP6\Predict\diff_2015@2090.csv"
    outdf.to_csv(outpath, index=False)