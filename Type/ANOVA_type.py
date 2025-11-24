import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')

import pandas as pd
import numpy as np
import scipy.stats as stats

import GlobVars as gv
import matplotlib.pyplot as plt

path = r"F:\Research\AMFEdge\EdgeType\anoVI_Amazon_specUndistEdge_effect_2023.csv"
df = pd.read_csv(path)
df = df[df['dNIRv_mean']<0]
dst_vars = ['NIRv']

# Perform ANOVA for each variable
for dst_var in dst_vars:
    col = 'd' + dst_var + '_mean'
    anova_data = df[['type', col]]
    anova_data = anova_data.dropna()

    # Perform ANOVA
    groups = [group[col].values for name, group in anova_data.groupby('type')]
    f_stat, p_value = stats.f_oneway(*groups)

    print(f"{dst_var} - F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
