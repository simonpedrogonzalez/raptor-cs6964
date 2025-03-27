import numpy as np
import geopandas as gpd

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

def vectorize(raster):
    """Vectorize a raster into a GeoDataFrame of points
    """

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

    return points

def get_polygon_bounding_box_pixels(raster, polygon):
    """Get the indices of the pixels intersected with
    the bounding box of a polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    row_min, col_min = raster.index(minx, maxy)
    row_max, col_max = raster.index(maxx, miny)
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(raster.height - 1, row_max)
    col_max = min(raster.width - 1, col_max)
    window = (col_min, row_min, col_max, row_max)

    return window


def read_raster_data(raster, mask, col_start=0, row_start=0):
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