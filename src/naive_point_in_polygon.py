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
from geocube.vector import vectorize
from shapely import points as shpoints
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("QtAgg")

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

def qsplit(raster, polygon, min_size=1e6):
    window = (0, 0, raster.width-1, raster.height-1)
    polygon = gpd.GeoDataFrame(geometry=[polygon], crs=raster.crs)
    s, c = qsplitr(raster, polygon, min_size, window)
    return s / c

def qsplitr(raster, polygon, min_size, window):
    
    n_cols, n_rows = window[2] - window[0], window[3] - window[1]
    
    # Completely outside case
    row_indices = np.array([window[1], window[3]])
    col_indices = np.array([window[0], window[2]])
    x, y = xy_np(raster.transform, row_indices, col_indices)
    w = box(x[0], y[0], x[1], y[1])
    if not w.intersects(polygon.iloc[0]).iloc[0]:
        print(f"Window: {window}, data: 0")
        return np.nan, 0

    # At least partially inside case
    if n_cols * n_rows <= min_size:
        row_indices, col_indices = np.meshgrid(np.arange(window[1], window[3]), np.arange(window[0], window[2]), indexing='ij')
        row_indices = row_indices.ravel()
        col_indices = col_indices.ravel()
        x, y = xy_np(raster.transform, row_indices, col_indices)
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=raster.crs)
        indices = naive_point_in_polygon(points, polygon)
        # indices = qsplit_point_in_polygon(points, polygon, 0, 3)
        if len(indices) == 0:
            print(f"Window: {window}, data: 0, --")
            return np.nan, 0
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        row_indices_in_window = row_indices[indices] - window[1]
        col_indices_in_window = col_indices[indices] - window[0]
        mask[row_indices_in_window, col_indices_in_window] = True
        data = read_data(raster, mask, window[0], window[1])
        print(f"Window: {window}, data: {len(data)}")
        return data.sum(), len(data)

    # Split case
    new_n_cols = n_cols // 2
    new_n_rows = n_rows // 2
    vertical_cut_point = window[0] + new_n_cols
    horizontal_cut_point = window[1] + new_n_rows
    windows = [
        (window[0], window[1], vertical_cut_point, horizontal_cut_point),
        (vertical_cut_point, window[1], window[2], horizontal_cut_point),
        (window[0], horizontal_cut_point, vertical_cut_point, window[3]),
        (vertical_cut_point, horizontal_cut_point, window[2], window[3])
    ]
    partial_sum = 0
    partial_count = 0
    for i, w in enumerate(windows):
        m_, c_ = qsplitr(raster, polygon, min_size, w)
        if np.isnan(m_):
            continue
        partial_sum += m_
        partial_count += c_

    return partial_sum, partial_count
    
def zonal_mean(raster, polygon, method="naive"):
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

    # t0 = t.time()
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=crs)
    # t1 = t.time()
    # print(f"Points creation time GEODF: {t1-t0}")

    # create with shapely and then put in a geodataframe
    # t0 = t.time()
    # points = shpoints(x, y=y)
    # points = gpd.GeoDataFrame(geometry=points, crs=crs)
    # t1 = t.time()
    # print(f"Points creation time SHAPELY: {t1-t0}")


    polygon = gpd.GeoDataFrame(geometry=[polygon], crs=crs)

    # get points in polygon and create a 2d mask to apply to the raster
    if method == "naive":
        indices = naive_point_in_polygon(points, polygon)
    elif method == "qsplit":
        indices = qsplit_point_in_polygon(points, polygon, 0, 3)
    
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[row_indices[indices], col_indices[indices]] = True

    # somewhat efficient reading
    data = read_data(raster, mask)
    return data.mean()

def read_data(raster, mask, col_start=0, row_start=0):
    # Get the extreme non-masked values
    non_masked_indices = np.where(mask)

    top = non_masked_indices[0].min()  # Row start
    bottom = non_masked_indices[0].max() + 1  # Row end
    left = non_masked_indices[1].min()  # Col start
    right = non_masked_indices[1].max() + 1  # Col end

    # Fix the window to use (col_off, row_off, width, height)
    window = rio.windows.Window(left + col_start, top + row_start, right - left, bottom - top)

    # Read the correct window
    data = raster.read(1, window=window)

    # Ensure mask matches the read window
    mask2 = mask[top:bottom, left:right]  # Fix slicing order

    # Correct index shift
    # data_indices_read = np.where(mask2)
    # data_indices_read = (data_indices_read[0] + top + row_start, data_indices_read[1] + left + col_start)

    return data[mask2]


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

t0 = t.time()
mean = qsplit(raster_layer, polygon, min_size=1e6)
t1 = t.time()
print(f"Qsplitv2 time: {t1-t0}")
print(f"Qsplitv2 value: {mean}")
assert np.isclose(true_mean, mean, atol=1e-6)

# Times:
# Rasterstats time: 5.2760491371154785 (But it calculates all the stats for all polys)
# True value: 301.062958996328
# Naive time: 0.5756285190582275
# Naive value: 301.0629577636719
# Qsplit time: 1.034510850906372 # Maybe it's useful for higher resolution rasters
# Qsplit value: 301.0629577636719

# read src data US_MSR.tif as raster and create a box polygon as the left half of the raster
# US MAP test
print("US MAP test")
raster_layer_file = f'{base}/US_MSR.tif'
raster_layer = rio.open(raster_layer_file)
bounds = raster_layer.bounds
polygon = box(bounds.left, bounds.bottom, (bounds.left + bounds.right)/2, bounds.top)
polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_layer.crs)

# Stats with rasterstats
t0 = t.time()
stats = zonal_stats(
    polygon_gdf,
    raster_layer_file,
    stats=["mean"],
    geojson_out=False
)
true_mean = stats[0]['mean']
t1 = t.time()
print(f"Rasterstats time: {t1-t0}")
print(f"True value: {true_mean}")

# qsplit method
t0 = t.time()
mean2 = qsplit(raster_layer, polygon, min_size=10e6)
t1 = t.time()
print(f"Qsplit time: {t1-t0}")
print(f"Qsplit value: {mean2}")
assert np.isclose(true_mean, mean2, atol=1e-6)
print('done')