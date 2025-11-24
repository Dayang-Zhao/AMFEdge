import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import linregress
import Attribution.LcoRF as lcorf

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)

def cm2inch(value):
    return value/2.54

def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def main(model, raw_df:pd.DataFrame, X:pd.DataFrame, features:list, grid:tuple, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(cm2inch(13), cm2inch(6.5))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            xcol, ycol, ccol, scol = features[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            xlabel = plot_setting['xlabels'][i*ncols+j]
            ylabel = plot_setting['ylabels'][i*ncols+j]
            xlim = plot_setting['xlims'][i*ncols+j]
            ylim = plot_setting['ylims'][i*ncols+j]
            cmap = plot_setting['cmaps'][i*ncols+j]
            norm = plot_setting['norms'][i*ncols+j]
            levels = plot_setting['levels'][i*ncols+j]

            # Plot.
            if j == 1:
                disp = PartialDependenceDisplay.from_estimator(
                    model, X, features=[xcol], kind="both", grid_resolution=50,
                    ax=ax, ice_lines_kw={'color':'#299d8f','linewidth':1, 'alpha':0.5, 'label':'Individual'},
                    pd_line_kw={'color':"#f3a361", 'linewidth':3, 'linestyle':'-', 'label':'Average'}
                )
                actual_ax = disp.axes_[0, 0]
                actual_ax.set_xlabel(xlabel, labelpad=-3)
                actual_ax.set_ylabel(ylabel)
                # Remove data distribution in xaxis.
                actual_ax.tick_params(which='minor', bottom=False, left=False)
                actual_ax.set_xlim(xlim)
                actual_ax.set_ylim(ylim)
                # Remove frames and ticks
                actual_ax.spines['top'].set_visible(False)
                actual_ax.spines['right'].set_visible(False)
                legend = actual_ax.legend(loc='upper left', frameon=False, fontsize=LABEL_SIZE)
            else:
                # raw_df = remove_outliers_iqr(raw_df, [xcol, ycol])
                ax.scatter(
                    raw_df[xcol], raw_df[ycol], c=raw_df[ccol], s=40,
                    marker='o', edgecolors='black', alpha=0.8, 
                    cmap=sns.color_palette("RdBu_r", as_cmap=True),
                    norm=mcolors.BoundaryNorm(boundaries=np.arange(-8, 9, 2), ncolors=cmap.N)
                )
                result = linregress(raw_df[xcol], raw_df[ycol])
                x = np.linspace(xlim[0], xlim[1], 100)
                y = result.intercept + result.slope * x
                ax.plot(x, y, color='black', linewidth=1.5, linestyle='--')
                ax.text(0.6, 0.25, '***$r$= '+str(result.rvalue.round(2)), fontsize=12, transform=ax.transAxes)
            
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
                ax.minorticks_off()
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(xlabel, labelpad=-3)
                ax.set_ylabel(ylabel)
                # Remove frames and ticks
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            ax.set_title(f"{title_num} {title}", fontsize=LABEL_SIZE+2, loc='left')
    
    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.98, hspace=0, wspace=0.3)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    # Train the model.
    xcols = ['HAND_mean', 'rh98_scale', 'rh98_magnitude', 
            'SCC_mean', 'sand_mean_mean', 'MCWD_mean', 'histMCWD_mean',
            'surface_solar_radiation_downwards_sum_mean',
            'vpd_mean', 'total_precipitation_sum_mean','temperature_2m_mean']
    ycol = 'nirv_scale'

    path = r"F:\Research\AMFEdge\Model\Amazon_undistEdge_Attribution.csv"
    raw_df = pd.read_csv(path)
    df = raw_df.dropna(subset=xcols+[ycol])
    df['MCWD_mean'] = df['MCWD_mean']*-1
    df['nirv_scale'] = df['nirv_scale']/1000
    df['rh98_scale'] = df['rh98_scale']/1000
    df = df[(df['nirv_scale'] <= 6)]
    X = df[xcols]
    y = df[ycol]

    model = lcorf.LcoRF()
    model.fit(X, y)

    # Test the model 
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R^2: {r2:.4f}")
    mse = mean_squared_error(y, y_pred)

    # Plot
    df = df[(df['rh98_scale'] <= 6)]
    # df = df[df['nirv_magnitude']>0]
    grid = (1,2)
    features = [('rh98_scale', 'nirv_scale','nirv_magnitude', 'nirv_scale'),]*2
    cmap1 = sns.color_palette(palette='RdBu_r', as_cmap=True)
    levels = np.arange(-8, 9, 2)
    norm1 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap1.N)
    norm2 = mcolors.BoundaryNorm(boundaries=np.arange(-4, 5, 1), ncolors=cmap1.N)
    cmaps = [cmap1, cmap1]
    norms = [norm1, norm2]

    plot_setting = {
        'titles': ['Observed', 'Modelled'],
        'xlabels': ['$S_{\mathrm{RH98}}$ (km)']*2,
        'ylabels': [r'$S_{\Delta \mathrm{NIRv}}$ (km)']*2,
        'xlims': [[0, 6.5], [0.5, 6]],
        'ylims': [[0, 6.5], [0.5, 6]],
        'cmaps': cmaps,'norms': norms, 'levels': [levels]*2
    }
    outpath = r'E:\Thesis\AMFEdge\Figures\Cause\Snirv_RH98.pdf'
    main(model, df, X, features, grid, plot_setting, outpath=outpath)
