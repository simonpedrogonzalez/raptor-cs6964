'''Attempt at Naive Point-in-Polygon method compared to most common methods'''

import geopandas as gpd
from rasterstats import zonal_stats
from geocube.vector import vectorize # couldnt make it work yet
import rioxarray as rxr
import rasterio as rio
from shapely.geometry import shape, Point, MultiPoint
from rasterio.features import shapes
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import time as t


base = 'src/data'
vector_layer_file = f'{base}/ne_10m_admin_0_countries.shp'
vector_layer = gpd.read_file(vector_layer_file)
raster_layer_file = f'{base}/ECMWF_utci_20230101_v1.1_con.nc'
raster_layer = rio.open(raster_layer_file)

# Here should go some checks, like checking both layers are using same CRS

# Stats with rasterstats
# I think it uses the masking method, that is
# it rasterizes the vector layer and then masks
# the raster layer with the rasterized vector layer
# and then calculates the stats
# Manual: https://pythonhosted.org/rasterstats/manual.html
# This would be the "to beat" method using raptor methods
# also, this method calculates the stats for all the polygons
# at once.
t0 = t.time()
stats = zonal_stats(
    vector_layer,
    raster_layer_file,
    stats=["mean"],
    geojson_out=False
)
true_mean = stats[0]['mean']
t1 = t.time()
print(f"Rasterstats time: {t1-t0}")
print(f"True value: {true_mean}")


def to_numpy2(transform):
    return np.array([transform.a, 
    transform.b, 
    transform.c, 
    transform.d, 
    transform.e, 
    transform.f, 0, 0, 1], dtype='float64').reshape((3,3))

def xy_np(transform, cols, rows, offset='center'):
    # https://gis.stackexchange.com/questions/415062/how-to-speed-up-rasterio-transform-xy
    # A faster xy trasnform than rasterio.transform.xy
    if isinstance(rows, int) and isinstance(cols, int):
        pts = np.array([[rows, cols, 1]]).T
    else:
        assert len(rows) == len(cols)
        pts = np.ones((3, len(rows)), dtype=int)
        pts[0] = rows
        pts[1] = cols

    if offset == 'center':
        coff, roff = (0.5, 0.5)
    elif offset == 'ul':
        coff, roff = (0, 0)
    elif offset == 'ur':
        coff, roff = (1, 0)
    elif offset == 'll':
        coff, roff = (0, 1)
    elif offset == 'lr':
        coff, roff = (1, 1)
    else:
        raise ValueError("Invalid offset")

    _transnp = to_numpy2(transform)
    _translt = to_numpy2(transform.translation(coff, roff))
    locs = _transnp @ _translt @ pts
    return locs[0], locs[1]

def naive_point_in_polygon(raster, polygon):
    # If the TODOs are solved, this should be equivalent
    # to the Naive PIP method shown in the paper.
    # Naive Point-in-Polygon
    # 1. Vectorize the raster layer (proyect pixels to vector space)
    # https://gis.stackexchange.com/questions/431918/vectorizing-a-raster-containing-holes-using-rasterio
    # 2. Check point in polygon for each pixel
    # 3. Recover values and compute mean

    transform = raster.transform
    crs = raster.crs

    points = []
    n_cols = raster.width
    n_rows = raster.height
    row_indices, col_indices = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')
    row_indices = row_indices.ravel()
    col_indices = col_indices.ravel()

    x, y = xy_np(transform, row_indices, col_indices)
    xy_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=crs)
    # get points in polygon and create a 2d mask to apply to the raster
    # TODO: the polygon.contains is super slow, it is actually the only slow part of the algo
    # we could try another library like xarray or geocube
    mask = polygon.contains(xy_gdf).values.ravel().reshape((n_rows, n_cols))
    # here is the problem, I should  be able to read the non masked values,
    # but it seems I can only read the whole raster and then mask it
    # TODO: test other reading libraries that allows reading non masked values
    data = raster.read(1)
    mean = data[mask].mean()
    return mean

def qtree_point_in_polygon(raster, polygon):
    # https://stackoverflow.com/questions/36282306/is-there-way-to-optimize-speed-of-shapely-geometry-shape-containsa-point-call
    # https://gis.stackexchange.com/questions/23688/splitting-vector-into-equal-smaller-parts-using-qgis/23694#23694
    # https://gis.stackexchange.com/questions/120955/understanding-use-of-spatial-indexes-with-rtree/144764#144764
    return None

first_element = vector_layer.iloc[0]
polygon = first_element.geometry
t0 = t.time()
mean = naive_point_in_polygon(raster_layer, polygon)
t1 = t.time()
print(f"Naive time: {t1-t0}")
print(f"Naive value: {mean}")
assert np.isclose(true_mean, mean, atol=1e-6)
print(f"Element: {first_element['SOVEREIGNT']}")
print('done')