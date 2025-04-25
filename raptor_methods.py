from zone_stat_method import ZonalStatMethod
from utils import vectorize_raster_to_points, split_window_into_quadrants
from rasterstats.io import Raster
import rasterio as rio
import rasterio.mask as rio_mask
from rasterio.features import geometry_window
import geopandas as gpd
from shapely import Geometry, box
import numpy as np
import shapely as shp
from shapely import Polygon, Point, LineString
from typing import List, Tuple, Dict
from rtree import index
from raster_methods import Masking
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Scanline(ZonalStatMethod):

    __name__ = "Scanline"

    def __init__(self):
        super().__init__()
    
    def _compute_scanline_intersections(self, y: float, x0: float, x1: float, feature: gpd.GeoDataFrame) -> List[float]:
        
        geom = feature.geometry[0]

        # Create a horizontal line and get intersections
        scanline = LineString([Point(x0, y), Point(x1, y)])
        
        if not scanline.intersects(geom):
            return []

        intersection = scanline.intersection(geom)

        # Handle different types of intersections
        if intersection.geom_type == 'Point':
            return [intersection.x]
        elif intersection.geom_type == 'MultiPoint':
            return sorted([p.x for p in intersection])
        elif intersection.geom_type == 'LineString':
            return sorted([intersection.coords[0][0], intersection.coords[-1][0]])
        elif intersection.geom_type == 'MultiLineString':
            xs = []
            for line in intersection.geoms:
                xs.extend([line.coords[0][0], line.coords[-1][0]])
            return sorted(xs)
        raise ValueError("Unexpected intersection geometry")
    
    def _process_scanline(self, y: float, intersections: List[float], raster: rio.DatasetReader) -> Dict[str, float]:
        """Returns the 1 row statistics for the scanline intersections.
        """

        row_results = []

        # Process pairs of intersections
        for i in range(0, len(intersections), 2):
            if i + 1 >= len(intersections):
                break

            # Convert start and end points to pixel coordinates
            x0, x1 = intersections[i], intersections[i + 1]
            
            # Start and end cols in pixel indexes relative to the raster
            row, start_col = rio.transform.rowcol(raster.transform, x0, y)
            _, end_col = rio.transform.rowcol(raster.transform, x1, y)

            if 0 <= row < raster.height:
                # Process pixels between intersections
                # Reading only the row of interest in between the intersections
                row_window = rio.windows.Window(
                    col_off=start_col,
                    row_off=row,
                    width=end_col - start_col,
                    height=1
                )
                data = raster.read(1, window=row_window, masked=True)
                intersection_result = self._compute_stats_from_masked_array(data)
                row_results.append(intersection_result)

        return self._combine_stats(row_results)

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):


        from raster_methods import Masking
        self.true_mask = Masking().mask(feature, raster, window)

        row_start, row_end = window.row_off, window.row_off + window.height
        col_start, col_end = window.col_off, window.col_off + window.width

        # Limits for the horizontal y scanlines, 1 pixel wider than the window
        x0 = (raster.transform * (col_start - 1, 0))[0]
        x1 = (raster.transform * (col_end, 0))[0]

        results = []

        for row in range(row_start, row_end + 1):
            
            y = (raster.transform * (0, row))[1]
            intersections = self._compute_scanline_intersections(y, x0, x1, feature)

            if len(intersections) >= 2:
                # Process the scanline
                result = self._process_scanline(row, y, intersections, raster, window)
                results.append(result)

        return self._combine_stats(results)



class Node():
    def __init__(self, _id, parent_id, level, _box, stats, max_depth):
        self.id = _id
        self.parent_id = parent_id
        self.level = level
        self.box = _box
        self.stats = stats
        self.max_depth = max_depth

    def is_contained_in_geom(self, geom: Geometry) -> bool:
        return self.box.within(geom)
    
    def is_leaf(self) -> bool:
        return self.level == self.max_depth

    def is_root(self) -> bool:
        return self.level == 0

class AggQuadTree(Scanline):

    
    __name__ = "AggQuadTree"

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        super().__init__()
    
    def _precomputations(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader):
        
        # width = raster.width
        # height = raster.height

        boxes_per_level = {}
        x_min, y_min, x_max, y_max = raster.bounds
        width = x_max - x_min
        height = y_max - y_min

        n_total_boxes = sum([4 ** i for i in range(self.max_depth + 1)])
        box_index = 0
        idx = index.Index()

        print(f"Total boxes: {n_total_boxes}")
        print(f"Width: {width}, Height: {height} in coordinates")
        print(f"Minx: {x_min}, Miny: {y_min} in coordinates")
        print(f"Maxx: {x_max}, Maxy: {y_max} in coordinates")
        print(f"Raster size: {raster.width} x {raster.height}")

        for level in range(self.max_depth, -1, -1):
            divisions = 2 ** level
            dx = width / divisions
            dy = height / divisions
            x_starts = np.linspace(x_min, x_max - dx, divisions)
            y_starts = np.linspace(y_min, y_max - dy, divisions)

            xs, ys = np.meshgrid(x_starts, y_starts)

            xs = xs.flatten()
            ys = ys.flatten()

            boxes = np.stack([xs, ys, xs + dx, ys + dy], axis=1)
            
            if level != 0:
                blocks = boxes.reshape(divisions//2, 2, divisions//2, 2, 4)\
                    .transpose(0, 2, 1, 3, 4)\
                    .reshape(-1, 4)
                boxes = blocks
                
            n_boxes = len(blocks)

            print(f"Level {level}: {n_boxes} boxes")
            print(f"Divisions: {divisions}")
            print(f"Len boxes: {len(boxes)}")
            print(f"max boxes: {max(boxes[:, 2])}, {max(boxes[:, 3])}")
            print(f"min boxes: {min(boxes[:, 0])}, {min(boxes[:, 1])}")
            print(f"First box: {boxes[0]}")

            ids = (np.arange(n_boxes) + box_index).astype('uint8')
            box_index += n_boxes
            parent_ids = (np.arange(n_boxes // 4) + box_index).astype('uint8').repeat(4)
            if level == 0:
                parent_ids = np.array([np.nan])

            shp_boxes1 = box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
            shp_boxes = [box(x0, y0, x1, y1) for x0, y0, x1, y1 in boxes]
            print(shp_boxes[0])
            print(shp_boxes1[0])
            print(boxes[0])
            if level == self.max_depth:
                
                vector_layer = gpd.GeoDataFrame(
                    geometry=shp_boxes,
                    crs=feature.crs
                )

                # count number of pixels in first box
                box1 = vector_layer.geometry[0]
                win = rio.windows.from_bounds(
                    box1.bounds[0],
                    box1.bounds[1],
                    box1.bounds[2],
                    box1.bounds[3],
                    transform=raster.transform
                )
                data = raster.read(window=win, masked=True)
                data = data[0]
                data = data[~data.mask]
                print(f"Data shape: {data.shape}")

                masking_method = Masking()
                aggregates = masking_method(raster, vector_layer, self.stats)
                aggregates = np.array(aggregates)
                print(f"Aggregates count: {aggregates[0]['count']}")
            else:
                # use self._combine_stats to combine the stats of the previous level
                # run combine over the previous_aggregates using previous_ids
                # to know which previous_aggregates belong to which parent

                aggregates = []
                for i in range(len(boxes)):
                    child_ids = previous_ids[previous_parent_ids == previous_parent_ids[i]] \
                        - previous_ids.min()
                    child_aggregates = previous_aggregates[child_ids]

                    aggregates.append(self._combine_stats(child_aggregates))
                
                aggregates = np.array(aggregates)

            previous_aggregates = aggregates
            previous_ids = ids
            previous_parent_ids = parent_ids
            
            # test_plot(boxes, aggregates)

            for i in range(len(boxes)):
                node = Node(
                    _id=ids[i],
                    parent_id=parent_ids[i],
                    level=level,
                    max_depth=self.max_depth,
                    _box=shp_boxes[i],
                    stats=aggregates[i]
                )
                idx.insert(ids[i], boxes[i], obj=node)

        self.idx = idx

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):

        # Get all intersecting nodes, sorted from root to leaves
        window_bounds = rio.windows.bounds(window, raster.transform)
        nodes = list(self.idx.intersection(window_bounds, objects=True))[::-1]
        nodes = [node.object for node in nodes]
        geom = feature.geometry[0]

        # Debugging code
        # global_mask = np.zeros((window.height, window.width), dtype=bool)

        partials = []
        while len(nodes) > 0:
            node = nodes.pop(0)
            if node.is_contained_in_geom(geom):
                print(f"Fully conained node")

                # add the entire node to the mask
                win = rio.windows.from_bounds(
                    *node.box.bounds,
                    transform=raster.transform
                )
                # Calculate the overlap of the node's window relative to the main processing window
                # row_start = int(win.row_off - window.row_off)
                # row_end   = int(row_start + win.height)
                # col_start = int(win.col_off - window.col_off)
                # col_end   = int(col_start + win.width)

                # # Update global mask
                # global_mask[row_start:row_end, col_start:col_end] = True
                # i want here to set true in the global_mask, which
                # is inside the window
                # end of debugging code

                partials.append(node.stats)
                if node.is_leaf():
                    continue
                # remove all children from the list
                for n in nodes:
                    if n.parent_id == node.id:
                        nodes.remove(n)
            else:
                if node.is_leaf():
                    print(f"Not contained leaf")
                    # is leaf but only intersects the geom
                    intersection_geom = node.box.intersection(geom)
                    if intersection_geom.is_empty:
                        # means that both the node intersects
                        # the window but not the feature
                        continue
                    win = rio.windows.from_bounds(
                        *intersection_geom.bounds,
                        transform=raster.transform
                    )
                    out_shape = (int(np.ceil(win.height)), int(np.ceil(win.width)))
                    mask = rio.features.rasterize(
                        [(intersection_geom, 1)],
                        out_shape=(int(np.ceil(win.height)), int(np.ceil(win.width))),
                        transform=rio.windows.transform(win, raster.transform),
                        fill=0,
                        dtype='uint8'
                    ).astype(bool)
                    data = raster.read(window=win, out_shape=out_shape, masked=True)[0]
                    data = data[mask]
                    partials.append(self._compute_stats_from_masked_array(data))

                    # add the mask to the global mask
                    
                    # map into global mask
                    # row_start = int(win.row_off - np.ceil(window.row_off))
                    # col_start = int(win.col_off - np.ceil(window.col_off))
                    # h, w = mask.shape
                    # global_mask[row_start:row_start + h, col_start:col_start + w] = mask
                    # assert global_mask.shape[0] >= row_start + h
                    # assert global_mask.shape[1] >= col_start + w

                    # end of debugging code

                else:
                    # not completely inside but has children, so the children
                    # will be analyzed in next iterations
                    continue
        
        # check global mask
        # plot_masks(raster, feature, global_mask, window)

        # combine the partials
        return self._combine_stats(partials)

def test_plot(boxes, aggregates):

    nx, ny = 8, 8
    # normalize boxes to fit in the figure
    x_min = min(boxes[:, [0, 2]].flatten())
    x_max = max(boxes[:, [0, 2]].flatten())
    y_min = min(boxes[:, [1, 3]].flatten())
    y_max = max(boxes[:, [1, 3]].flatten())

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_min) / (x_max - x_min) * nx
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_min) / (y_max - y_min) * ny



    fig, ax = plt.subplots(figsize=(nx, ny))

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # Label the box with its index at the center
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        tt = aggregates[i]['count']
        ax.text(cx, cy, tt, fontsize=8, ha='center', va='center')

    ax.set_xlim(min(boxes[:, 0]), max(boxes[:, 2]))
    ax.set_ylim(min(boxes[:, 1]), max(boxes[:, 3]))
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Optional: to mimic raster image top-down layout
    plt.grid(True)
    plt.savefig('test_plot.png')


def plot_masks(raster, geom, my_mask, window):

    correct_mask = Masking().mask(geom, raster, window)
    # plot 3 things:
    # my_mask
    # correct_mask
    # difference between the two masks
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(my_mask, cmap='gray')
    ax[0].set_title('My Mask')
    ax[1].imshow(correct_mask, cmap='gray')
    ax[1].set_title('Correct Mask')
    ax[2].imshow(my_mask.astype(int) - correct_mask.astype(int), cmap='gray')
    ax[2].set_title('Difference')
    # add colorbar to the first two plots
    
    # plot difference between the two masks
    # mask - correct_mask
    print('done')

    


def test_raster():

    # 10x10 raster with all 1s
    array = np.ones((8, 8), dtype=np.uint8)
    transform = rio.transform.from_origin(0, 8, 1, 1)  # (x_min, y_max, x_res, y_res)
    from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH


    # Create the GeoTIFF file
    with rio.open(
        f"{RASTER_DATA_PATH}/test.tif",
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs="EPSG:4326",  # WGS84
        transform=transform,
    ) as dst:
        dst.write(array, 1)

def test():

    from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH

    ag = AggQuadTree(max_depth=3)
    vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
    raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR.tif'

    # test_raster()
    # raster_layer_file = f'{RASTER_DATA_PATH}/test.tif'
    ag(raster_layer_file, vector_layer_file, ['count', 'mean'])



# test()