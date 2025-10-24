import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import GlobVars as gv

def cal_total_frac(df):
    pos_gain = df['pos_edge_gain'].sum() + df['pos_int_gain'].sum()
    neg_gain = df['neg_edge_gain'].sum() + df['neg_int_gain'].sum()
    net_gain = pos_gain + neg_gain

    # Net loss.
    edge_gain_frac = (df['pos_edge_gain'].sum()+df['neg_edge_gain'].sum())/net_gain
    int_gain_frac = (df['pos_int_gain'].sum()+df['neg_int_gain'].sum())/net_gain

    # Gross loss.
    pos_edge_gain_frac = df['pos_edge_gain'].sum()/pos_gain
    pos_int_gain_frac = df['pos_int_gain'].sum()/pos_gain
    neg_edge_gain_frac = df['neg_edge_gain'].sum()/neg_gain
    neg_int_gain_frac = df['neg_int_gain'].sum()/neg_gain

    # Count.
    pos_count = df['pos_edge_count'].sum() + df['pos_int_count'].sum()
    neg_count = df['neg_edge_count'].sum() + df['neg_int_count'].sum()
    pos_edge_count_frac = df['pos_edge_count'].sum()/pos_count
    pos_int_count_frac = df['pos_int_count'].sum()/pos_count
    neg_edge_count_frac = df['neg_edge_count'].sum()/neg_count
    neg_int_count_frac = df['neg_int_count'].sum()/neg_count

    outdf = pd.DataFrame({
        'edge_gain_frac': edge_gain_frac,
        'int_gain_frac': int_gain_frac,
        "pos_edge_gain_frac": pos_edge_gain_frac,
        "neg_edge_gain_frac": neg_edge_gain_frac,
        "pos_int_gain_frac": pos_int_gain_frac,
        "neg_int_gain_frac": neg_int_gain_frac,
        "pos_edge_count_frac": pos_edge_count_frac,
        "pos_int_count_frac": pos_int_count_frac,
        "neg_edge_count_frac": neg_edge_count_frac,
        "neg_int_count_frac": neg_int_count_frac
    }, index=[0])

    return outdf

def monte_carlo_R(df, nmc=5000, random_state=None):
    df = df.reset_index(drop=True)
    df = df.dropna(subset=["rh98_scale", "NIRv_count", "NIRv_mean_ano", "NIRv_mean_raw"])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(30)
    n = len(df)

    # 批量采样：shape = (nmc, n)
    nirv_ano_samples = rng1.normal(
        loc=df["NIRv_mean_ano"].values,
        scale=df["NIRv_stdDev_ano"].values,
        size=(nmc, n)
    )
    nirv_raw_samples = rng2.normal(
        loc=df["NIRv_mean_raw"].values,
        scale=df["NIRv_stdDev_raw"].values,
        size=(nmc, n)
    )

    # 固定参数（广播用）
    nirv_count = df["NIRv_count"].values
    rh98_scale = df["rh98_scale"].values
    dist = df["Dist"].values

    # mask：edge 和 interior
    edge_mask = (dist <= rh98_scale) & (dist != -1)
    int_mask  = (dist == -1) | (dist > rh98_scale)

    # 计算每次模拟的forests count.
    pos_edge_count = ((nirv_ano_samples>0)*nirv_count)[:, edge_mask].sum(axis=1)
    pos_int_count  = ((nirv_ano_samples>0)*nirv_count)[:, int_mask].sum(axis=1)
    neg_edge_count = ((nirv_ano_samples<0)*nirv_count)[:, edge_mask].sum(axis=1)
    neg_int_count  = ((nirv_ano_samples<0)*nirv_count)[:, int_mask].sum(axis=1)
    total_pos_count = pos_edge_count + pos_int_count
    total_neg_count = neg_edge_count + neg_int_count
    pos_edge_count_frac = pos_edge_count / total_pos_count
    neg_edge_count_frac = neg_edge_count / total_neg_count
    pos_int_count_frac = pos_int_count / total_pos_count
    neg_int_count_frac = neg_int_count / total_neg_count

    # 计算每次模拟的gross loss
    pos_ano = np.where(nirv_ano_samples>0, nirv_ano_samples, 0)/100
    neg_ano = np.where((nirv_ano_samples<0)&(nirv_ano_samples > -100), nirv_ano_samples, 0)/100
    pos_edge_gain = (pos_ano * nirv_raw_samples * nirv_count)[:, edge_mask].sum(axis=1)
    pos_int_gain  = (pos_ano * nirv_raw_samples * nirv_count)[:, int_mask].sum(axis=1)
    neg_edge_gain = (neg_ano * nirv_raw_samples * nirv_count)[:, edge_mask].sum(axis=1)
    neg_int_gain  = (neg_ano * nirv_raw_samples * nirv_count)[:, int_mask].sum(axis=1)
    pos_gain = pos_edge_gain + pos_int_gain
    neg_gain = neg_edge_gain + neg_int_gain
    pos_gain_frac = pos_gain/(nirv_raw_samples * nirv_count).sum(axis=1)
    neg_gain_frac = neg_gain/(nirv_raw_samples * nirv_count).sum(axis=1)

    pos_edge_gain_frac = pos_edge_gain / pos_gain
    neg_edge_gain_frac = neg_edge_gain / neg_gain
    pos_int_gain_frac = pos_int_gain / pos_gain
    neg_int_gain_frac = neg_int_gain / neg_gain

    # 计算每次模拟的net loss
    edge_loss = (nirv_ano_samples * nirv_raw_samples * nirv_count)[:, edge_mask].sum(axis=1)
    int_loss  = (nirv_ano_samples * nirv_raw_samples * nirv_count)[:, int_mask].sum(axis=1)
    total_loss = edge_loss + int_loss
    edge_loss_frac = edge_loss / total_loss
    int_loss_frac = int_loss / total_loss

    # 拼成 DataFrame
    outdf = pd.DataFrame({
        "num": np.arange(nmc),
        "pos_gain_frac": pos_gain_frac,
        "neg_gain_frac": neg_gain_frac,
        'edge_loss_frac': edge_loss_frac,
        'int_loss_frac': int_loss_frac,
        "pos_edge_gain_frac": pos_edge_gain_frac,
        "neg_edge_gain_frac": neg_edge_gain_frac,
        "pos_int_gain_frac": pos_int_gain_frac,
        "neg_int_gain_frac": neg_int_gain_frac,
        "pos_edge_count_frac": pos_edge_count_frac,
        "pos_int_count_frac": pos_int_count_frac,
        "neg_edge_count_frac": neg_edge_count_frac,
        "neg_int_count_frac": neg_int_count_frac
    })

    # outdf = outdf[(outdf['edge_loss_frac'] > 0) & (outdf['edge_loss_frac'] < 1) & (outdf['int_loss_frac'] > 0) & (outdf['int_loss_frac'] < 1)].reset_index(drop=True)
    return outdf


def cal_nirv_loss(df):
    ids = df['Id'].unique()
    outdf = pd.DataFrame(columns=['Id', "pos_edge_count",
                    "pos_int_count", "neg_edge_count",
                    "neg_int_count", "pos_edge_gain",
                    "pos_int_gain", 'neg_edge_gain',
                    'neg_int_gain',])

    for id in ids:
        dst_df = df.loc[df['Id'] == id]
        int_df = dst_df.loc[(dst_df['Dist']==-1)|(dst_df['Dist']>dst_df['rh98_scale'])]
        edge_df = dst_df.loc[(dst_df['Dist']<=dst_df['rh98_scale'])&(dst_df['Dist']!=-1)]

        edge_gain = (edge_df['NIRv_mean_ano']*edge_df['NIRv_mean_raw']*edge_df['NIRv_count'])
        int_gain = (int_df['NIRv_mean_ano']*int_df['NIRv_mean_raw']*int_df['NIRv_count'])
        pos_edge_gain = edge_gain[edge_gain > 0].sum()
        neg_edge_gain = edge_gain[edge_gain < 0].sum()
        pos_int_gain = int_gain[int_gain > 0].sum()
        neg_int_gain = int_gain[int_gain < 0].sum()

        # Net loss.
        net_gain = edge_gain.sum() + int_gain.sum()
        edge_gain_frac = edge_gain.sum()/net_gain
        int_gain_frac = int_gain.sum()/net_gain

        # Gross loss.
        pos_edge_gain_frac = pos_edge_gain / (pos_edge_gain + pos_int_gain)
        pos_int_gain_frac = pos_int_gain / (pos_edge_gain + pos_int_gain)
        neg_edge_gain_frac = neg_edge_gain / (neg_edge_gain + neg_int_gain)
        neg_int_gain_frac = neg_int_gain / (neg_edge_gain + neg_int_gain)

        # Count.
        total_count = int_df['NIRv_count'].sum() + edge_df['NIRv_count'].sum()
        pos_count = dst_df[dst_df['NIRv_mean_ano'] > 0]['NIRv_count'].sum()
        neg_count = dst_df[dst_df['NIRv_mean_ano'] < 0]['NIRv_count'].sum()
        pos_edge_count = edge_df[edge_df['NIRv_mean_ano'] > 0]['NIRv_count'].sum()
        pos_int_count = int_df[int_df['NIRv_mean_ano'] > 0]['NIRv_count'].sum()
        neg_edge_count = edge_df[edge_df['NIRv_mean_ano'] < 0]['NIRv_count'].sum()
        neg_int_count = int_df[int_df['NIRv_mean_ano'] < 0]['NIRv_count'].sum()
        pos_edge_count_frac = pos_edge_count / pos_count
        pos_int_count_frac = pos_int_count / pos_count
        neg_edge_count_frac = neg_edge_count / neg_count
        neg_int_count_frac = neg_int_count / neg_count


        out_row = {'Id': id, 
                   "pos_edge_count": pos_edge_count,
                    "pos_int_count": pos_int_count,
                    "neg_edge_count": neg_edge_count,
                    "neg_int_count": neg_int_count, 
                    "pos_edge_gain": pos_edge_gain,
                    "pos_int_gain": pos_int_gain,
                    'neg_edge_gain': neg_edge_gain,
                    'neg_int_gain': neg_int_gain,
        }
        outdf.loc[len(outdf)] = out_row

    outdf2 = cal_total_frac(outdf)

    return outdf2

if __name__ == "__main__":
    anoVI_path = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_Edge_2023.csv"
    anoVI_df = pd.read_csv(anoVI_path)

    VI_path = r"F:\Research\AMFEdge\EdgeVI\VI_Amazon_Edge_2023.csv"
    VI_df = pd.read_csv(VI_path)

    num_path = r"F:\Research\AMFEdge\EdgeNum\anoVI_Amazon_Edge_2023.csv"
    num_df = pd.read_csv(num_path)

    rh_path = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_Effect_2023.csv"
    rh_df = pd.read_csv(rh_path)

    merged_df = anoVI_df[['Id', 'Dist', 'NIRv_mean', 'NIRv_stdDev']].merge(VI_df[['Id', 'Dist', 'NIRv_mean', 'NIRv_stdDev']], on=['Id', 'Dist'], suffixes=('_ano', '_raw'))
    merged_df = merged_df.merge(num_df[['Id', 'Dist', 'NIRv_count']], on=['Id', 'Dist'])
    merged_df = merged_df.merge(rh_df[['Id', 'rh98_scale']], on='Id')
    merged_df = merged_df.dropna(subset=['rh98_scale'], axis=0)
    # merged_df = merged_df[merged_df['NIRv_mean_ano'] < 0]
    # merged_df = merged_df[merged_df['Id'].isin(gv.SWGRID_IDS)]

    outdf = cal_nirv_loss(merged_df)
    # outdf = monte_carlo_R(merged_df, nmc=10000, random_state=42)

    outpath = r"F:\Research\AMFEdge\EdgeVI\NIRvLoss_Amazon_Edge_2023.csv"
    # outpath = r"F:\Research\AMFEdge\EdgeVI\NIRvLossFrac_Monte_Amazon_Edge_2023.csv"
    outdf.to_csv(outpath, index=False)
    print(outdf)