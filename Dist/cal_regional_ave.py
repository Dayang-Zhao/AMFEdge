import pandas as pd
import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import GlobVars as gv

dst_var = 'ndwi'

def cal_regional_ave(df:pd.DataFrame, dst_col:str):
    df = df.loc[df[dst_col].notna(),:]
    df['class'] = df['Id'].isin(gv.NEGRID_IDS).map({True: 1, False: 2})
    df = df.loc[df[dst_var+'_scale']<=6000]

    # Select NE and SW grids.
    ne_data = df.loc[df['class']==1, dst_col].dropna()
    sw_data = df.loc[df['class']==2, dst_col].dropna()

    def _cal_mean_std(data):
        q5 = data.quantile(0.05)
        q95 = data.quantile(0.95)
        filtered = data[(data >= q5) & (data <= q95)]
        mean_val = filtered.mean()
        std_val = filtered.std()
        return mean_val, std_val

    ne_mean, ne_std = _cal_mean_std(ne_data)
    sw_mean, sw_std = _cal_mean_std(sw_data)

    print(f"NE Mean and std (5th-95th percentile): {ne_mean, ne_std}")
    print(f"SW Mean and std (5th-95th percentile): {sw_mean, sw_std}")

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\Edge\Amazon_UndistEdge_Effect_2023.csv"
    df = pd.read_csv(path)
    cal_regional_ave(df, dst_col=dst_var+'_magnitude')