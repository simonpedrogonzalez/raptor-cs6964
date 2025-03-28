from rasterstats import zonal_stats
import geopandas as gpd
import rasterio as rio
import numpy as np

def reference_method(raster_file_path, vector_file_path, stats):
    vector = gpd.read_file(vector_file_path)
    zs = zonal_stats(vector, raster_file_path, stats=stats, boundless=True)

    # return one dict for each polygon [ {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'count': 1}, ... ]
    return zs

def compare_results(result1, result2, allow_None=True, **kwargs):
    # per polugon
    for i in range(len(result1)):
        # per stat
        if allow_None and (result1[i] is None or result2[i] is None):
            continue
        
        for key in result1[i].keys():
            value1 = result1[i][key]
            value2 = result2[i][key]

            if not np.isclose(value1, value2, **kwargs):
                return False, {
                    "index": i,
                    "key": key,
                    "value1": result1[i][key],
                    "value2": result2[i][key]
                }
    return True, {}