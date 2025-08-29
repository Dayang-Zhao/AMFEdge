import pandas as pd

SEP_VAR = 'rh98'

def cal_interior_value(df:pd.DataFrame, dst_vars:list) -> pd.DataFrame:

    # Calculate interior forest value.
    outdf = pd.DataFrame(columns=['Id'] + [f'{var}_int' for var in dst_vars])
    ids = df['Id'].unique()
    for id in ids:
        dst_df = df.loc[df['Id'] == id]

        # Interior forest value.
        int_df = dst_df.loc[dst_df['Dist']>=dst_df[f'{SEP_VAR}_scale']]
        int_values = {}
        for dst_var in dst_vars:
            int_value = (int_df[f'{dst_var}_mean']*int_df[f'{dst_var}_count']).sum()/ int_df[f'{dst_var}_count'].sum()
            int_values[dst_var+'_int'] = int_value

        out_row = {'Id': id} | int_values
        outdf.loc[len(outdf)] = out_row

    return outdf

def main(VI_df: pd.DataFrame, RH_df:pd.DataFrame, dst_vars:list):
    
    merged_df = pd.merge(VI_df, RH_df, on='Id')

    # Calculate interior forest value.
    int_df = cal_interior_value(merged_df, dst_vars)

    # Calculate difference in VI between interior and edge.
    merged_df = pd.merge(merged_df, int_df, on='Id', how='left')

    for dst_var in dst_vars:
        merged_df[f'{dst_var}_diff'] = merged_df[f'{dst_var}_mean'] - merged_df[f'{dst_var}_int']

    return merged_df

if __name__ == '__main__':
    # Read data.
    VI_path = r"F:\Research\AMFEdge\Edge\anoVI_Amazon_UndistEdge_2023.csv"
    VI_df = pd.read_csv(VI_path)
    RH_path = r"F:\Research\AMFEdge\EdgeRH\Amazon_UndistEdge_Effect_2023.csv"
    RH_df = pd.read_csv(RH_path)
    dst_vars = ['NIRv', 'EVI', 'NDWI']

    merged_df = main(VI_df, RH_df, dst_vars)

    outpath = r"F:\Research\AMFEdge\EdgeRH\dAnoVI_Amazon_UndistEdge_2023.csv"
    merged_df.to_csv(outpath, index=False)
