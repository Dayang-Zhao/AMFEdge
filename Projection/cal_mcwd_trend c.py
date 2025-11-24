import pandas as pd
from scipy.stats import linregress

import matplotlib.pyplot as plt

ycol = 'relAnoMCWD_mean'

def linreg(group):
    res = linregress(group["year"], group[ycol])
    return pd.Series({"slope": res.slope, "p": res.pvalue})

def main(df):
    df = df.dropna(subset=[ycol])
    result = df.groupby(["Id",'model']).apply(linreg).reset_index()
    return result

if __name__ == "__main__":
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']

    outdfs = []
    for scenario in scenarios:
        df = pd.read_csv(rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_Edge_pred_{scenario}_V3.csv")
        # df2 = df.drop(columns=['model'])
        # df2 = df2.groupby(['Id', 'year']).mean().reset_index()
        df['MCWD_mean'] = df['MCWD_mean']*100/df['histMCWD_mean']
        outdf = main(df)
        outdf['scenario'] = scenario
        outdfs.append(outdf)

    outdf = pd.concat(outdfs, ignore_index=True)

    outpath = r"F:\Research\AMFEdge\CMIP6\Predict\MCWD_trend.csv"
    outdf.to_csv(outpath, index=False)