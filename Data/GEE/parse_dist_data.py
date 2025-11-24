import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import ast

import matplotlib.pyplot as plt
plt.ion()

# Convert GEE data of varying distances.
def parse_dist_data(df:pd.DataFrame, dist_columns:list, info_columns:list=['Id', 'system:index', 'Continent', 'GEZcode', 'Tot'],
                    other_columns: list=['anoMCWD'])->pd.DataFrame:
    """ The raw dataframe:
        system:index                                                 -1  ...                                     anoMCWD                                               .geo
    0            87       [5850, 1.813811063967263, 3.361138929372706]  ...  [-3.6644577585165994, 0.22685251056432784]  {"type":"Polygon","coordinates":[[[-66.9259565...
    1            88   [2250567, 0.8670188586469533, 3.258204789988136]  ...  [-3.7979022031866903, 0.14368481943666458]  {"type":"Polygon","coordinates":[[[-66.9259565...
    2            89    [2941612, 2.210653184878587, 1.861990600236643]  ...  [-3.6114265081597843, 0.22794739705540082]  {"type":"Polygon","coordinates":[[[-66.9259565...
    3            97  [5356046, -0.5074982868596087, 3.4968424042240...  ...  [-3.5457173038503917, 0.16078008982868294]  {"type":"Polygon","coordinates":[[[-65.6269178...
    4            98   [5336246, 1.6665715154114376, 2.970810949370067]  ...    [-3.628886592531846, 0.2623615465228132]  {"type":"Polygon","coordinates":[[[-65.6269178...

    [5 rows x 158 columns]
    is converted into:
          Id system:index Continent  GEZcode       Tot                                               .geo  aveAnoMCWD  stdAnoMCWD   Dist      Mean       Std    Count
    0    279           87  Americas     11.0  479376.0  {"type":"Polygon","coordinates":[[[-66.9259565...   -3.664458    0.226853     -1  1.813811  3.361139   5850.0
    1    279           87  Americas     11.0  479376.0  {"type":"Polygon","coordinates":[[[-66.9259565...   -3.664458    0.226853    120  1.725433  5.280918    600.0
    2    279           87  Americas     11.0  479376.0  {"type":"Polygon","coordinates":[[[-66.9259565...   -3.664458    0.226853    240  2.069001  3.568777    631.0
    3    279           87  Americas     11.0  479376.0  {"type":"Polygon","coordinates":[[[-66.9259565...   -3.664458    0.226853    360  1.516071  3.757672    425.0
    4    279           87  Americas     11.0  479376.0  {"type":"Polygon","coordinates":[[[-66.9259565...   -3.664458    0.226853    480  1.464488  3.419704    360.0
    ..   ...          ...       ...      ...       ...                                                ...         ...         ...    ...       ...       ...      ...
    750  301           98  Americas     11.0  394089.0  {"type":"Polygon","coordinates":[[[-65.6269178...   -3.628887    0.262362  17520  1.540339  2.974811  21706.0
    751  301           98  Americas     11.0  394089.0  {"type":"Polygon","coordinates":[[[-65.6269178...   -3.628887    0.262362  17640  1.553754  3.009331  22372.0
    752  301           98  Americas     11.0  394089.0  {"type":"Polygon","coordinates":[[[-65.6269178...   -3.628887    0.262362  17760  1.487179  2.920287  22872.0
    753  301           98  Americas     11.0  394089.0  {"type":"Polygon","coordinates":[[[-65.6269178...   -3.628887    0.262362  17880  1.434461  2.842630  22773.0
    754  301           98  Americas     11.0  394089.0  {"type":"Polygon","coordinates":[[[-65.6269178...   -3.628887    0.262362  18000  1.381436  2.811328  22776.0

    [755 rows x 12 columns]
    """
    # Extract columns.
    dists = list(set(info_columns) ^ set(df.columns)^ set(other_columns))
    outdf = pd.DataFrame(columns=info_columns+other_columns+dist_columns)
    for row_num in range(len(df)):
        row = df.iloc[row_num,:]
        row_df = pd.DataFrame(columns=info_columns+dist_columns)
        row_df['Dist'] = [int(x) for x in dists]

        # Extract data for each distance.
        for n, var in enumerate(dist_columns[1:]):
            row_df[var] = [ast.literal_eval(row[x].replace('null', '0'))[n] 
                           if ast.literal_eval(row[x].replace('null', '0'))[0]!=0 else np.nan 
                           for x in dists]

        # Extract other columns.
        row_df[info_columns+other_columns] = row.loc[info_columns+other_columns]
        row_df = row_df.sort_values(by='Dist')
        outdf = pd.concat([outdf, row_df], axis=0, ignore_index=True)

    return outdf

if __name__ == '__main__':
    # Read data.
    import os
    import glob

    year = 2023
    root_dir = r"F:\Research\AMFEdge\Edge1deg\metaData"
    pre_fname = f"anoVI_Amazon_Edge_{year}_"
    paths = glob.glob(pre_fname+'*'+'.csv', root_dir=root_dir)
    os.chdir(root_dir)
    # info_colums = ['Id', 'system:index', 'Continent', 'GEZcode', 'Tot']
    info_columns = ['Id']
    raw_df_list = []
    for path in paths:
        df = pd.read_csv(path)
        raw_df_list.append(df.drop(columns=['.geo', 'system:index', 'bottom','left','right','top','TreeCover'])
                             .set_index(info_columns)
                        )
    raw_df = pd.concat(raw_df_list, axis=1).reset_index()
    # raw_df = raw_df.drop(columns=['660', '780', '900'])
    
    # Parse data.
    other_columns = ['MCWD_mean','MCWD_stdDev', 'anoMCWD_mean', 'anoMCWD_stdDev']
    dist_columns = ['Dist',
        'EVI_count', 'EVI_max', 'EVI_mean', 'EVI_median', 'EVI_skew', 'EVI_stdDev', 'EVI_sum',
        'NDWI_count', 'NDWI_max', 'NDWI_mean', 'NDWI_median', 'NDWI_skew', 'NDWI_stdDev', 'NDWI_sum',
        'NIRv_count', 'NIRv_max', 'NIRv_mean', 'NIRv_median', 'NIRv_skew', 'NIRv_stdDev', 'NIRv_sum',
        ]
    # stats = ['count', 'max', 'mean', 'median', 'skew', 'stdDev', 'sum']
    # dist_columns = ['Dist'] + ['undistForest_'+x for x in stats] + ['deforestLand_'+x for x in stats]\
    #     + ['degradedForest_'+x for x in stats] + ['regrowthForest_'+x for x in stats]
    # dist_columns = ['Dist'] + ['rh50_'+x for x in stats] + ['rh98_'+x for x in stats]
    df2 = parse_dist_data(df=raw_df, dist_columns=dist_columns, 
                          info_columns=info_columns, other_columns=other_columns)

    # Export reshaped data.
    outpath = rf"F:\Research\AMFEdge\Edge1deg\anoVI_Amazon_Edge_{year}.csv"
    # outpath = r"F:\Research\AMFEdge\Edge_TreeCover\treeCover_Amazon_UndistEdge_2023.csv"
    df2.to_csv(outpath, index=False)