from zone_stat_method import ZonalStatMethod
from naive_method import NaivePointInPolygon
from utils import vectorize_raster_to_points, split_window_into_quadrants
from rasterstats.io import Raster
import rasterio as rio
from rasterio.features import geometry_window
import geopandas as gpd
from shapely import Geometry
import numpy as np
import shapely as shp

class QSplit(NaivePointInPolygon):

    __name__ = "QSlit"

    def __init__(self, min_size=1e6):
        """Class to compute the zonal statistics using the QSplit method.

        Parameters
        ----------
        min_size : int, optional
            Split the raster in quadrants until the size of the quadrant is less than this value.
            The default is 1e6.
        """
        self.min_size = min_size
        super().__init__()

    def _qsplit_recursive(self, raster: rio.DatasetReader, feature: gpd.GeoDataFrame, window: rio.windows.Window):
    
        n_cols, n_rows = window.width, window.height
        
        # Completely outside case
        window_box = shp.box(*rio.windows.bounds(window, raster.transform))
        if not window_box.intersects(feature.iloc[0].geometry):
            return None
        
        # At least partially inside case
        if n_cols * n_rows <= self.min_size:
            # Minimum size reached
            return self._naive_point_in_polygon(feature, raster, window)

        # Split case
        windows = split_window_into_quadrants(window)

        partial_results = [
            self._qsplit_recursive(raster, feature, w)
            for w in windows
        ]

        return self._combine_stats(partial_results)

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):

        points, pixel_indices = vectorize_raster_to_points(raster, window)

        return self._qsplit_recursive(raster, feature, window)