import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
from scipy.stats import linregress

import numpy as np
import pandas as pd
from functools import reduce
from sklearn.ensemble import RandomForestRegressor
import scipy.sparse
from sklearn.metrics import mean_squared_error, r2_score

def to_array(self):
    return self.toarray()

scipy.sparse.spmatrix.A = property(to_array)
from pygam import LinearGAM, s, f, te

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.ion()
import GlobVars as gv

xcols = ['length_mean', 'HAND_mean','undistForest_dist', 
        'SCC_mean', 'sand_mean_mean',
        'anoMCWD_mean','surface_net_solar_radiation_sum_mean',
        'vpd_mean', 'total_precipitation_sum_mean'
        ]
LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

def main(datas:list, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(12.5), cm2inch(6.5))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            y, y_pred, r2, mse = datas[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            xlabel = plot_setting['xlabels'][i*ncols+j]
            ylabel = plot_setting['ylabels'][i*ncols+j]
            xlim = plot_setting['xlims'][i*ncols+j]
            ylim = plot_setting['ylims'][i*ncols+j]

            # Plot.
            ax.scatter(y, y_pred, s=20, alpha=0.6)
            ax.axline((0, 0), (1, 1), linewidth=1.5, color='r')
            
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
            ax.minorticks_off()
            ax.set_title(f"{title_num} {title}", fontsize=LABEL_SIZE+2, loc='left')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.text(0.05, 0.8, f'$R^2$ = {r2:.3f}\nMSE = {mse**0.5:.3f}',
                    transform=ax.transAxes, fontsize=LABEL_SIZE+2)
    
    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.98, hspace=0, wspace=0.25)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)


if __name__ == "__main__":
    path = r"F:\Research\AMFEdge\Model\Amazon_Attribution.csv"
    df = pd.read_csv(path)
    df['nirv_scale'] = df['nirv_scale']/1000
    ycols = ['nirv_magnitude', 'nirv_scale']
    dst_df = df[xcols+ycols].dropna(axis=0)

    datas = []
    for ycol in ycols:
        X = dst_df[xcols].values
        y = dst_df[ycol].values

        X = dst_df[xcols].values
        y = dst_df[ycol].values

        model = RandomForestRegressor(n_estimators=50, max_depth=5, criterion='squared_error',
                                    min_samples_split=2, min_samples_leaf=2, random_state=42)
        model.fit(X, y)
        # Predict
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        datas.append((y, y_pred, r2, mse))

    # Plot setting
    plot_setting = {
        'titles': ['$\Delta$NIRv Magnitude (%)', '$\Delta$NIRv Scale (km)'],
        'xlabels': ['Predicted values (%)', 'Predicted values (km)'],
        'ylabels': ['Observed values (%)', 'Observed values (km)'],
        'xlims': [[-6, 6], [0, 6.1]],
        'ylims': [[-6, 6], [0, 6.1]]
    }
    outpath = r"E:\Thesis\AMFEdge\Figures\Model\RF_performance.jpg"
    main(datas, grid=(1, 2), plot_setting=plot_setting, outpath=outpath)
