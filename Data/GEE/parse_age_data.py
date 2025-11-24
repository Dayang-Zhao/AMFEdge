import numpy as np
import pandas as pd
import ast

# # Convert GEE data of varying distances.
# def parse_dist_data(df:pd.DataFrame, dst_vars:list, stat_columns:list)->pd.DataFrame:
    
#     def parse_row(row, dst_var):
#         dst_value = ast.literal_eval(row[dst_var].replace('null', '0'))
#         for stat_column in stat_columns:
#             if dst_value[0] == 0:
#                 row[dst_var + '_' + stat_column] = np.nan
#             else:
#                 row[dst_var + '_' + stat_column] = dst_value[stat_columns.index(stat_column)]
#         return row

#     outrows = []
#     for dst_var in dst_vars:
#         for i in range(len(df)):
#             row = df.iloc[i]
#             row = parse_row(row, dst_var)
#             outrows.append(row.to_frame().T)
#     outdf = pd.concat(outrows, axis=0, ignore_index=True)

#     return outdf
def parse_age_data(df: pd.DataFrame, dst_vars: list, stat_columns: list) -> pd.DataFrame:
    new_data = {}
    
    for dst_var in dst_vars:
        parsed = df[dst_var].str.replace('null', '0').apply(ast.literal_eval)
        parsed = parsed.apply(lambda x: x if x[0] != 0 else [np.nan] * len(stat_columns))
        parsed_df = pd.DataFrame(parsed.tolist(), columns=[f"{dst_var}_{stat}" for stat in stat_columns])
        new_data.update(parsed_df.to_dict(orient='series'))

    df_cleaned = pd.concat([df.drop(columns=dst_vars), pd.DataFrame(new_data)], axis=1)
    return df_cleaned

if __name__ == '__main__':
    # Read data.
    import os
    import glob

    root_dir = r"F:\Research\AMFEdge\EdgeAge\metaData"
    edge_type = 'sum'
    pre_fname = f'anoVI_Amazon_{edge_type}UndistEdge_2023_y'
    paths = glob.glob(pre_fname+'*'+'.csv', root_dir=root_dir)
    os.chdir(root_dir)
    info_colums = ['system:index','TreeCover','bottom','left','right','top','Id']
    raw_df_list = []
    for path in paths:
        # age = path.split('_')[-1].split('.')[0][1:]
        age = (int(path.split('_')[-1].split('.')[0][1:]) + int(path.split('_')[-2][1:]))/2
        df = pd.read_csv(path)
        df['age'] = age
        raw_df_list.append(df.drop(columns=['.geo'])
                             .set_index(info_colums)
                        )
    raw_df = pd.concat(raw_df_list, axis=0).reset_index()
    
    # Parse data.
    dst_vars = ['NIRv', 'EVI', 'NDWI']
    stats = ['count', 'max', 'mean', 'median', 'skew', 'stdDev', 'sum']

    df2 = parse_age_data(df=raw_df, dst_vars=dst_vars, stat_columns=stats)

    # Export reshaped data.
    # outpath = r"F:\Research\AMFEdge\EdgeAge\sumEdge\anoVI_Amazon_sumUndistEdge_2023_age.xlsx"
    # with pd.ExcelWriter(outpath, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    #     df2.to_excel(writer, sheet_name=edge_type, index=False)
    outpath = r"F:\Research\AMFEdge\EdgeAge\anoVI_Amazon_sumUndistEdge_2023_age.csv"
    df2.to_csv(outpath, index=False)