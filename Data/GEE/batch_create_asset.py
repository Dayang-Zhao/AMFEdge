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
dists = np.arange(60, 301, 60).tolist() + np.arange(420, 901, 120).tolist() \
    + np.arange(1080, 1801, 180).tolist() + np.arange(2040, 3001, 240).tolist() \
    + np.arange(3300, 4501, 300).tolist()+ np.arange(4860, 6301, 360).tolist()
    
    # + np.arange(8880, 10801, 480).tolist()+ np.arange(6720, 8401, 420).tolist()
    # + np.arange(11340, 13501, 540).tolist()+ np.arange(14100, 16501, 600).tolist()
# dists = np.arange(1000, 6001, 1000).tolist()
parent_path = "projects/forestedge-432402/assets/AMFSumEdge2023/"
asset_ids = ['Edge'+ str(dist) for dist in dists]
batch_create_asset(parent_path, asset_ids)