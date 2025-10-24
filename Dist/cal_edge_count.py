import pandas as pd

def cal_edge_area(num_df:pd.DataFrame, rh_df:pd.DataFrame) -> pd.DataFrame:
    """Calculate area of edge and interior forest."""

    df = num_df.merge(rh_df[['Id', 'rh98_scale']], on='Id')

    # Calculate area of interior and edge forest.
    outdf = pd.DataFrame(columns=['Id', 'edge_count', 'interior_count'])
    ids = df['Id'].unique()
    for id in ids:
        dst_df = df.loc[df['Id'] == id]
        int_df = dst_df.loc[(dst_df['Dist']==-1)|(dst_df['Dist']>dst_df['rh98_scale'])]
        edge_df = dst_df.loc[(dst_df['Dist']<=dst_df['rh98_scale'])&(dst_df['Dist']!=-1)]

        out_row = {'Id': id, 'edge_count': edge_df['NIRv_count'].sum(), 'interior_count': int_df['NIRv_count'].sum()}
        outdf.loc[len(outdf)] = out_row

    outdf['edge_frac'] = outdf['edge_count']/(outdf['edge_count']+outdf['interior_count'])
    outdf['interior_frac'] = outdf['interior_count']/(outdf['edge_count']+outdf['interior_count'])
    return outdf

if __name__ == '__main__':
    # Read data.
    num_path = r"F:\Research\AMFEdge\EdgeNum\anoVI_Amazon_Edge_2023.csv"
    num_df = pd.read_csv(num_path)

    rh98_path = r"F:\Research\AMFEdge\EdgeRH\Amazon_Edge_Effect_2023.csv"
    rh98_df = pd.read_csv(rh98_path)

    area_df = cal_edge_area(num_df, rh98_df)

    outpath = r"F:\Research\AMFEdge\EdgeNum\Area_Amazon_Edge_2023.csv"
    area_df.to_csv(outpath, index=False)
