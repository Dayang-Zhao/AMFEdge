import os
import subprocess
from datetime import datetime

import pandas as pd

import ee
ee.Authenticate()
ee.Initialize(project='forestedge-432402')

MODELS = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
            'noresm2_mm', 'miroc6', 'taiesm1',
            'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']
VARS = ['total_evaporation_sum','surface_solar_radiation_downwards_sum', 
        'total_precipitation_sum', 'temperature_2m', 
         'vpd', 'WD']
STD_TIME = pd.date_range(start='2015-01-01', end='2100-12-31', freq='MS').strftime('%Y%m%d').tolist()
# STD_TIME = pd.date_range(start='1985-01-01', end='2014-12-31', freq='MS').strftime('%Y%m%d').tolist()
# Setting.
experiment = 'SSP2_45'  # 'Hist', 'SSP1_26', 'SSP2_45', 'SSP5_85'
local_rootdir = rf"F:\Research\AMFEdge\CMIP6\Processed\QDM\{experiment}"
gs_rootdir = f"gs://gee_amfedge_bucket/CMIP6/{experiment}"   # 你的tif文件夹路径
asset_folder = rf"projects/forestedge-432402/assets/CMIP6/{experiment}"  # 目标Asset目录
# 'gs://gee_amfedge_bucket/CMIP6/SSP5_85/mri_esm2_0_total_evaporation_sum.tif'
# Upload tif files to Google Earth Engine.
for model in MODELS:
    for nvar in VARS:
        fname = f"{model}_{nvar}.tif"
        gs_path = f"{gs_rootdir}/{model}_{nvar}.tif"

        # Generate band names.
        bands = [f"{nvar}_{date}" for date in STD_TIME]

        date_str = '2015-05-01'

        # Asset ID.
        asset_id = f"{asset_folder}/{model}/{nvar}"

        # earthengine manifest command.
        bands = [
            {'id': band, 'tilesetBandIndex': i, 
             'pyramidingPolicy': 'MEAN', } 
            for i, band in enumerate(bands)
            ]
        manifest = {
            "name": asset_id,
            "tilesets": [
                {
                    "sources": [
                        {
                            "uris": [gs_path]
                        }
                    ]
                }
            ],
            "bands": bands,
            "startTime": date_str + "T00:00:00Z"
        }

        print(f"Uploading {gs_path} to {asset_id} ...")
        request_id = ee.data.newTaskId()[0]
        task_id = ee.data.startIngestion(request_id=request_id, params=manifest)
