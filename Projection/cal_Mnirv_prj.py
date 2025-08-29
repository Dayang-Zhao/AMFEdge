import pandas as pd

import matplotlib.pyplot as plt
plt.ion()

def cal_ave_M(df):
    # Calculate the area-weighted M.
    posdf = df[df['nirv_magnitude'] > 0]
    negdf = df[df['nirv_magnitude'] < 0]
    pos_ave = (posdf['nirv_magnitude']*posdf['edge_count']).sum()/posdf['edge_count'].sum()
    neg_ave = (negdf['nirv_magnitude']*negdf['edge_count']).sum()/negdf['edge_count'].sum()
    outrow = {'pos_ave':pos_ave, 'neg_ave':neg_ave}

    return outrow

if __name__ == "__main__":
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']
    area_path = r"F:\Research\AMFEdge\EdgeNum\Area_Amazon_UndistEdge_2023.csv"
    area_df = pd.read_csv(area_path)
    outdf = pd.DataFrame(columns=['scenario', 'pos_ave', 'neg_ave'])
    for scenario in scenarios:
        path = rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_pred_{scenario}.csv"
        df = pd.read_csv(path)
        df = df[df['year']>=2030]
        df = df.merge(area_df, on='Id')
        outdf_row = cal_ave_M(df)
        outdf_row['scenario'] = scenario
        outdf.loc[len(outdf)] = outdf_row

    outpath = r"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_pred_ave.csv"
    outdf.to_csv(outpath, index=False)