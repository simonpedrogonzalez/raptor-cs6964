'''Attempt at Naive Point-in-Polygon method compared to most common methods'''

import geopandas as gpd
from rasterstats import zonal_stats
from geocube.vector import vectorize # couldnt make it work yet
import rioxarray as rxr
import rasterio as rio
from shapely.geometry import shape, Point, MultiPoint, box
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

def zonal_mean(raster, polygon, method="naive"):
    # If the TODOs are solved, this should be equivalent
    # to the Naive PIP method shown in the paper.
    # Naive Point-in-Polygon
    # 1. Vectorize the raster layer (proyect pixels to vector space)
    # https://gis.stackexchange.com/questions/431918/vectorizing-a-raster-containing-holes-using-rasterio
    # 2. Check point in polygon for each pixel
    # 3. Recover values and compute mean

    transform = raster.transform
    crs = raster.crs

    # It's not clear if I should take the height and width of the raster or
    # the bounding box of the polygon to create the meshgrid
    n_cols = raster.width
    n_rows = raster.height
    
    row_indices, col_indices = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')
    row_indices = row_indices.ravel()
    col_indices = col_indices.ravel()

    x, y = xy_np(transform, row_indices, col_indices)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=crs)

    polygon = gpd.GeoDataFrame(geometry=[polygon], crs=crs)

    # get points in polygon and create a 2d mask to apply to the raster
    if method == "naive":
        indices = naive_point_in_polygon(points, polygon)
    elif method == "qsplit":
        indices = qsplit_point_in_polygon(points, polygon, 0, 3)
    
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[row_indices[indices], col_indices[indices]] = True

    # here is the problem, I should  be able to read the non masked values,
    # but it seems I can only read the whole raster and then mask it
    # TODO: test other reading libraries that allows reading non masked values
    data = raster.read(1)
    mean = data[mask].mean()
    return mean

def naive_point_in_polygon(points: GeoDataFrame, polygon: GeoDataFrame):
    indices = gpd.tools.sjoin(points, polygon, predicate='within', how='inner').index
    return indices

def qsplit_point_in_polygon(points: GeoDataFrame, polygon: GeoDataFrame, depth: int, max_depth: int, crs=None):
    # https://stackoverflow.com/questions/36282306/is-there-way-to-optimize-speed-of-shapely-geometry-shape-containsa-point-call
    # https://gis.stackexchange.com/questions/23688/splitting-vector-into-equal-smaller-parts-using-qgis/23694#23694
    # https://gis.stackexchange.com/questions/120955/understanding-use-of-spatial-indexes-with-rtree/144764#144764
    # It would be same as above but in the PIP phase, it would
    # apply a quadtree to the polygon and then check the points

    if depth >= max_depth or len(points) <= 100:  # We might need to adjust this threshold
        polygon = gpd.GeoDataFrame(geometry=polygon, crs=crs)
        return gpd.tools.sjoin(points, polygon, predicate='within', how='inner').index

    minx, miny, maxx, maxy = polygon.total_bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    quadrants = [
        box(minx, cy, cx, maxy),
        box(cx, cy, maxx, maxy),
        box(minx, miny, cx, cy),
        box(cx, miny, maxx, cy) 
    ]

    # print(f"Depth: {depth}: boxes: {len(quadrants)}")
    
    indices = []
    for quad in quadrants:
        sub_poly = polygon.intersection(quad).dropna()
        if not sub_poly[0].is_empty:
            sub_points = points[points.geometry.within(quad)]
            indices.extend(qsplit_point_in_polygon(sub_points, sub_poly, depth + 1, max_depth, polygon.crs))

    return indices

first_element = vector_layer.iloc[0]
polygon = first_element.geometry
method = "naive"

t0 = t.time()
mean = zonal_mean(raster_layer, polygon, method)
t1 = t.time()
print(f"Naive time: {t1-t0}")
print(f"Naive value: {mean}")
assert np.isclose(true_mean, mean, atol=1e-6)

method = "qsplit"
t0 = t.time()
mean = zonal_mean(raster_layer, polygon, method)
t1 = t.time()
print(f"Qsplit time: {t1-t0}")
print(f"Qsplit value: {mean}")
assert np.isclose(true_mean, mean, atol=1e-6)

print(f"Element: {first_element['SOVEREIGNT']}")
print('done')

# Times:
# Rasterstats time: 5.2760491371154785 (But it calculates all the stats for all polys)
# True value: 301.062958996328
# Naive time: 0.5756285190582275
# Naive value: 301.0629577636719
# Qsplit time: 1.034510850906372 # Maybe it's useful for higher resolution rasters
# Qsplit value: 301.0629577636719