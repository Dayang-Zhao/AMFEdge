import os
import glob

import numpy as np
import pandas as pd
import ast

import matplotlib.pyplot as plt

root_dir = r"F:\Research\TropicalForestEdge\Test\NIRv\Hist"
pre_fname = 'Hist_Grid'

def read_data():
    os.chdir(root_dir)
    fnames = glob.glob(pre_fname+'*.csv', root_dir=root_dir)

    df_list = []
    for fname in fnames:
        sp_fname = fname.split('_')
        Id = int(sp_fname[1][4:])
        if sp_fname[2] == 'Intact.csv':
            Dist = -1
        else:
            Dist = int(sp_fname[2][:-5])
        temp_df = pd.read_csv(fname)
        temp_df['NIRv_ano Count'] = temp_df['NIRv_ano Count'].str.replace(',', '').astype(float)
        columns = temp_df.columns
        temp_df['Id'] = Id
        temp_df['Dist'] = Dist

        # Reindex columns.
        temp_df = temp_df[['Id', 'Dist']+columns.to_list()]
        # Calculate Percentage.
        total_count = temp_df['NIRv_ano Count'].sum()
        temp_df['Percentage'] = temp_df['NIRv_ano Count']*100/total_count

        df_list.append(temp_df)

    outdf = pd.concat(df_list, axis=0)
    
    return outdf

if __name__ == '__main__':
    outdf = read_data()
    outpath = r"F:\Research\TropicalForestEdge\Test\NIRv\Hist\Grid_Hist_sum.csv"
    outdf.to_csv(outpath, index=False)