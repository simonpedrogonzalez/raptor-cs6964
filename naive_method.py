from zonal_stat_method import ZonalStatMethod
from utils import vectorize

class NaivePointInPolygon(ZonalStatMethod):

    def __init__(self):
        pass

    def _read_vector_file(self):
        pass

    def _geometry_iterator(self):

        vector_layer = self._read_vector_file()
        def geom_iter():
            pass

        return geom_iter()

    def _compute(self):

        # maybe in area only
        points = vectorize(raster)

        for geom in self._geom_iter():
            
            indices = gpd.tools.sjoin(points, polygon, predicate='within', how='inner').index

            mask = np.zeros((n_rows, n_cols), dtype=bool)
            mask[row_indices[indices], col_indices[indices]] = True

            # somewhat efficient reading
            data = read_data(raster, mask)
            
            return data.mean()