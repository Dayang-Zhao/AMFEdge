import pandas as pd

def main(grid):
    path1 = r"F:\Research\AMFEdge\Edge\anoVI_panAmazon_UndistEdge_2023.xlsx"
    df1 = pd.read_excel(path1, sheet_name=grid)
    df1 = df1.loc[df1['Dist']<=6000]
    df1 = df1.rename(columns={'NIRv_mean': 'anoNIRv_mean', 'NDWI_mean': 'anoNDWI_mean',
                              'NIRv_mstd': 'anoNIRv_mstd', 'NDWI_mstd': 'anoNDWI_mstd',
                              'NIRv_count': 'anoNIRv_count', 'NDWI_count': 'anoNDWI_count'})
    
    path2 = r"F:\Research\AMFEdge\EdgeVI\VI_panAmazon_UndistEdge_DrySeason.xlsx"
    df2 = pd.read_excel(path2, sheet_name=grid)
    df2 = df2.loc[df2['Dist']<=6000]

    # Merge dataframes.
    merged_df = pd.merge(df1, df2, on=['Dist'], how='outer')
    loss = (merged_df['anoNIRv_mean'] * merged_df['NIRv_mean'] * merged_df['NIRv_count']).sum()
    ave_loss = loss / merged_df['NIRv_count'].sum()

    return ave_loss

if __name__ == '__main__':
    grid = 'SWGRID'
    ave_loss = main(grid)
    print(f"Average loss for {grid}: {ave_loss:.4f}")