import ee

# Initialize Earth Engine
# ee.Authenticate()
ee.Initialize(project='forestedge-432402')


def batch_copy_asset(srcParent, dstParent, id_start_with=''):
    # List assets in the specified parent folder or collection
    src_asset_paths = ee.data.listAssets({'parent': srcParent})['assets']

    for src_asset in src_asset_paths:
        src_asset_id = str(src_asset['name'])  # Get the asset ID as a string
        if src_asset_id.startswith(id_start_with):  # Check if it starts with the specified prefix
            dst_asset_id = src_asset_id.replace(srcParent, dstParent)
            dst_asset_id = dst_asset_id.replace("Intact2023", "Intact")
            ee.data.copyAsset(src_asset_id, dst_asset_id)  # Copy the asset
            print(f"Copied {src_asset_id} to {dst_asset_id}")

# ******** Test ********
srcParent = "projects/forestedge-432402/assets/ForestEdge2023/AmeEdgeIntact/"
dstParent = "projects/forestedge-432402/assets/AmazonForestEdge2023/EdgeIntact/"
id_start_with = "projects/forestedge-432402/assets/ForestEdge2023/AmeEdgeIntact/Intact2023"

batch_copy_asset(srcParent=srcParent, dstParent=dstParent, id_start_with=id_start_with)
