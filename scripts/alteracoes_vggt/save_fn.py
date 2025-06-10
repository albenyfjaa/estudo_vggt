import numpy as np
import rasterio

def save_depth_geotiff(depth_map, meta, output_path):
    """
    Save the depth map as a GeoTIFF using the metadata from the input file.

    Args:
        depth_map (np.array): 2D array representing the depth map.
        meta (dict): Metadata from the original GeoTIFF.
        output_path (str): Path to save the depth GeoTIFF.
    """
    # Update metadata for single band output and appropriate data type.
    meta.update({
        "count": 1,
        "dtype": 'float32'
    })
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(depth_map.astype(np.float32), 1)
