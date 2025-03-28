from zone_stat_method import ZonalStatMethod
from utils import vectorize_raster_to_points
from rasterstats.io import Raster
import rasterio as rio
from rasterio.features import geometry_window
import geopandas as gpd
from shapely import Geometry
import numpy as np

class NaivePointInPolygon(ZonalStatMethod):

    __name__ = "NaivePointInPolygon"

    def __init__(self):
        super().__init__()

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):
        return self._naive_point_in_polygon(feature, raster, window)

    def _naive_point_in_polygon(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):

        points, pixel_indices = vectorize_raster_to_points(raster, window)

        # Naive point in polygon check
        indexes = gpd.tools.sjoin(
            points,
            feature,
            predicate='within',
            how='inner'
            ).index

        # Relative mask to the window
        mask = np.zeros((window.height, window.width), dtype=bool)
        row_indices, col_indices = pixel_indices
        mask[row_indices[indexes] - window.row_off, col_indices[indexes] - window.col_off] = True
        
        data = raster.read(window=window, masked=True)
        data = data[:, mask]
        
        return self._compute_stats_from_masked_array(data)