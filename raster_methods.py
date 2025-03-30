from zone_stat_method import ZonalStatMethod
from utils import vectorize_raster_to_points, split_window_into_quadrants
from rasterstats.io import Raster
import rasterio as rio
import rasterio.mask as rio_mask
from rasterio.features import geometry_window
import geopandas as gpd
from shapely import Geometry
import numpy as np
import shapely as shp

class Masking(ZonalStatMethod):

    __name__ = "Masking"

    def __init__(self):
        super().__init__()

    def mask(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):
        """Create the mask used for the zonal statistics, useful for debugging.
        """
        mask = rio.features.rasterize(
            [(shp.geometry.mapping(feature.geometry[0]), 1)],
            out_shape=(window.height, window.width),
            transform=rio.windows.transform(window, raster.transform),
            fill=0,
            dtype='uint8'
        ).astype(bool)

        return mask

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):
        """Computes the zonal statistics for a feature using the Masking method.

        In this case, 1 feature, is virtually the same as the clipping method. The difference lays in the
        several features case where each feature has a different value in the mask.
        """
        
        mask = rio.features.rasterize(
            [(shp.geometry.mapping(feature.geometry[0]), 1)],
            out_shape=(window.height, window.width),
            transform=rio.windows.transform(window, raster.transform),
            fill=0,
            dtype='uint8'
        ).astype(bool)

        data = raster.read(window=window, masked=True)
        data = data[:, mask]
        return self._compute_stats_from_masked_array(data)


class Clipping(ZonalStatMethod):
    
    __name__ = "Clipping"

    def __init__(self):
        super().__init__()

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):

        # nodata_value = raster.nodata if raster.nodata is not None else -999

        data = raster.read(window=window, masked=True)
        clipped_data, _ = rio.mask.mask(raster, [feature.geometry[0]], crop=False, filled=False)
        
        return self._compute_stats_from_masked_array(clipped_data[0]) # Return first band
