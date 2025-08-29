import pandas as pd

SEP_VAR = 'rh98'

def cal_edge_area(df:pd.DataFrame, edge_dist:int=120, intact_dist:int=-1) -> pd.DataFrame:

    # Calculate area of interior and edge forest.
    outdf = pd.DataFrame(columns=['Id', 'edge_count', 'intact_count'])
    ids = df['Id'].unique()
    for id in ids:
        dst_df = df.loc[df['Id'] == id]
        int_df = dst_df.loc[dst_df['Dist']==intact_dist]
        edge_df = dst_df.loc[(dst_df['Dist']<=edge_dist)&(dst_df['Dist']>intact_dist)]

        out_row = {'Id': id, 'edge_count': edge_df['NIRv_count'].sum(), 'intact_count': int_df['NIRv_count'].sum()}
        outdf.loc[len(outdf)] = out_row

    return outdf

if __name__ == '__main__':
    # Read data.
    VI_path = r"F:\Research\AMFEdge\EdgeNum\anoVI_Amazon_UndistEdge_2023.csv"
    VI_df = pd.read_csv(VI_path)

    merged_df = cal_edge_area(VI_df)

    outpath = r"F:\Research\AMFEdge\EdgeNum\Area_Amazon_UndistEdge_2023.csv"
    merged_df.to_csv(outpath, index=False)
