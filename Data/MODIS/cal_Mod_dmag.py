import osgeo
import xarray as xr

import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

year = 2005
path = rf"F:\Research\AMFEdge\EdgeMod\anoVI_Amazon_UndistEdge_{year}.csv"
df = pd.read_csv(path)

pivot_df = df.pivot(index='Id', columns='Dist', values=['NIRv_mean', 'NDWI_mean'])

onset_dist = 1000
end_dists = [3000, 4000, 5000, 6000]

results = []
for end_dist in end_dists:
    result = pd.DataFrame({
        'Id': pivot_df.index,
        f'dNIRv_{int(onset_dist/100)}_{int(end_dist/100)}': pivot_df['NIRv_mean'][end_dist] - pivot_df['NIRv_mean'][onset_dist],
        f'dNDWI_{int(onset_dist/100)}_{int(end_dist/100)}': pivot_df['NDWI_mean'][end_dist] - pivot_df['NDWI_mean'][onset_dist]
    }).reset_index(drop=True)
    results.append(result)

outdf = reduce(lambda left, right: pd.merge(left, right, on='Id'), results)

outdf.to_csv(rf"F:\Research\AMFEdge\EdgeMod\anoVI_Amazon_UndistEdge_{year}_diff.csv", index=False)

print(outdf)