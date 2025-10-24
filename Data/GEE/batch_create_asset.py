import numpy as np
import ee

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='forestedge-432402')

def batch_create_asset(parent_path, asset_ids):

    assetMetadata = {'type': 'ImageCollection'}
    for asset_id in asset_ids:
        asset_full_path = parent_path + asset_id

        # Create the asset
        ee.data.createAsset(assetMetadata, asset_full_path)
        print(f"Created {asset_id}")

# ******** Test ********
# dists = np.arange(60, 301, 60).tolist() + np.arange(420, 901, 120).tolist() \
#     + np.arange(1080, 1801, 180).tolist() + np.arange(2040, 3001, 240).tolist() \
#     + np.arange(3300, 4501, 300).tolist()+ np.arange(4860, 6301, 360).tolist()
    
    # + np.arange(8880, 10801, 480).tolist()+ np.arange(6720, 8401, 420).tolist()
    # + np.arange(11340, 13501, 540).tolist()+ np.arange(14100, 16501, 600).tolist()
# dists = np.arange(1000, 6001, 1000).tolist()
parent_path = "projects/forestedge-432402/assets/CMIP6/"
subfolders = ['Hist', 'SSP1_26', 'SSP2_45', 'SSP5_85']
asset_ids = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
          'noresm2_mm', 'miroc6', 'taiesm1',
          'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']

for subfolder in subfolders:
    batch_create_asset(parent_path + subfolder + "/", asset_ids)