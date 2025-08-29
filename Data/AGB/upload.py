import os
import subprocess
from datetime import datetime

import ee
ee.Authenticate()
ee.Initialize(project='forestedge-432402')

BANDS = ['agb']

# Setting.
local_rootdir = r"F:\Research\AMFEdge\AGB\GlobBiomass\Amazon"
gs_rootdir = f"gs://gee_amfedge_bucket/GlobBiomass/Amazon"   # 你的tif文件夹路径
asset_folder = r"projects/forestedge-432402/assets/GlobBiomass/AMFBiomass"  # 目标Asset目录

# Upload tif files to Google Earth Engine.
for fname in os.listdir(local_rootdir):
    if fname.lower().endswith(".tif"):
        # gs_path = os.path.join(gs_rootdir, fname)
        gs_path = f"{gs_rootdir}/{fname}"

        # Extract date from filename
        year = 2010
        month = 1
        day = 1

        # Generate timestamp in milliseconds
        date_str = f"{year}-{month:02d}-{day:02d}"
        time_start = f"{year}-{month:02d}-{day:02d}"

        # Asset ID.
        asset_id = f"{asset_folder}/{fname[:-4]}"

        # earthengine manifest command.
        bands = [
            {'id': band, 'tilesetBandIndex': i, 
             'pyramidingPolicy': 'MEAN', } 
            for i, band in enumerate(BANDS)
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

        print(f"Uploading {fname} to {asset_id} ...")
        request_id = ee.data.newTaskId()[0]
        task_id = ee.data.startIngestion(request_id=request_id, params=manifest)
    # earthengine upload command.
    # cmd = [
    #     "earthengine", "upload", "image", "--asset_id", asset_id,
    #     "--pyramiding_policy", "MEAN",
    #     "--nodata_value", "-9999",
        #     "--time_start", time_start,
        #     "--bands", BANDS,
        #     gs_path
        # ]

        # print("Start uploading:", fname)
        # subprocess.run(cmd)
