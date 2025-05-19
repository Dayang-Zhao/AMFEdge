import numpy as np
import ee

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='forestedge-432402')


def batch_query_asset(parent, id_start_with):
    # List assets in the specified parent folder or collection
    asset_paths = ee.data.listAssets({'parent': parent})['assets']

    len_asset = 0
    for asset in asset_paths:
        asset_id = str(asset['name'])  # Get the asset ID as a string
        if asset_id.startswith(id_start_with):  # Check if it starts with the specified prefix
            len_asset += 1

    print(f'{parent} has {len_asset} assets')

# ******** Test ********
dists = np.arange(60, 301, 60).tolist() + np.arange(420, 901, 120).tolist() \
    + np.arange(1080, 1801, 180).tolist() + np.arange(2040, 3001, 240).tolist() \
    + np.arange(3300, 4501, 300).tolist()+ np.arange(4860, 6301, 360).tolist()\
    + np.arange(6720, 8401, 420).tolist()+ np.arange(8880, 10801, 480).tolist()\
    + np.arange(11340, 13501, 540).tolist()+ np.arange(14100, 16501, 600).tolist()
root_path = "projects/forestedge-432402/assets/AmazonForestEdge2023/"

for dist in dists:
    parent = root_path + 'Edge' + str(dist) + '/'
    id_start_with = root_path + 'Edge' + str(dist) + '/UndistEdge' + str(dist)
    batch_query_asset(parent, id_start_with)
