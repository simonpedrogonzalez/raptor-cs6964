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
from shapely import Polygon, Point, LineString, MultiLineString
from typing import List, Tuple, Dict
from rtree import index
from raster_methods import Masking
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import line_profiler

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
    
    def _process_scanline(
        self, y: float, intersections: List[float], raster: rio.DatasetReader,
        # mask: np.ndarray = None, row_shift=None, col_shift=None,# debugging
        ) -> Dict[str, float]:
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
                
                # Debugging code
                # mark the pixels in the global mask
                # rmask = np.zeros((1, end_col - start_col), dtype=bool)
                # rmask[0, :] = True
                # win_row = int(row - row_shift)
                # mask[win_row, start_col-col_shift:end_col-col_shift] = rmask[0, :]
                # end of debugging code

        return self._combine_stats(row_results) # , mask

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):


        # Debugging code
        # from raster_methods import Masking
        # self.true_mask = Masking().mask(feature, raster, window)

        row_start, row_end = window.row_off, window.row_off + window.height
        col_start, col_end = window.col_off, window.col_off + window.width

        # Limits for the horizontal y scanlines, 1 pixel wider than the window
        x0 = (raster.transform * (col_start - 1, 0))[0]
        x1 = (raster.transform * (col_end, 0))[0]

        # Debug code
        # global_mask = np.zeros((window.height, window.width), dtype=bool)

        results = []

        for row in range(row_start, row_end + 1):
            
            y = (raster.transform * (0, row))[1]
            intersections = self._compute_scanline_intersections(y, x0, x1, feature)

            if len(intersections) >= 2:

                # Process the scanline
                result = self._process_scanline(
                    y, intersections, raster,
                )

                # Debugging code
                # row_shift = row_start
                # col_shift = col_start
                # result, global_mask = self._process_scanline(
                #     y, intersections, raster,
                #     global_mask, row_shift,col_shift # debugging
                # )

                results.append(result)

        # Debugging code
        # plot_masks(raster, feature, global_mask, window)
        
        return self._combine_stats(results)



class Scanline2(ZonalStatMethod):

    __name__ = "Scanline2"

    def __init__(self):
        super().__init__()

    @line_profiler.profile
    def _precomputations2(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        

        transform = raster.transform

        def rows_to_ys(rows):
            return (transform * (0, rows + 0.5))[1]
        
        def rowcol_to_xy(rows, cols):
            # coords of pixel centers
            xs, ys = (transform * (cols + 0.5, rows + 0.5))
            return xs, ys

        def ys_to_rows(ys):
            return ((~transform) * (np.zeros_like(ys), ys))[1].astype(int)
        
        def xy_to_rowcol(xs, ys):
            # pixel in which the point is located
            cols, rows = (~transform) * (xs, ys)
            return np.floor(rows).astype(int), np.floor(cols).astype(int)

        # get window for the entire features, that is, the bounding box for all features
        window = rio.windows.from_bounds(
            *features.total_bounds,
            transform=transform
        )

        # Debugging code
        # w_height = int(np.ceil(window.height))
        # w_width = int(np.ceil(window.width))
        # global_mask = np.zeros((w_height+1, w_width), dtype=int)
        # End Debugging code

        row_start, row_end = int(np.floor(window.row_off)), int(np.ceil(window.row_off + window.height))
        col_start, col_end = int(np.floor(window.col_off)), int(np.ceil(window.col_off + window.width))

        x0 = (raster.transform * (col_start - 1, 0))[0]
        x1 = (raster.transform * (col_end, 0))[0]
        all_rows = np.arange(row_start, row_end + 1)
        ys = rows_to_ys(all_rows)
        
        x0s = np.full_like(ys, x0, dtype=float)
        x1s = np.full_like(ys, x1, dtype=float)

        # all points defining the scanlines
        all_points = np.stack([
            np.stack([x0s, ys], axis=1),  # shape (N, 2) -> start points
            np.stack([x1s, ys], axis=1)   # shape (N, 2) -> end points
        ], axis=1)  # shape (N, 2, 2)

        scanlines = MultiLineString(list(all_points))
        all_intersections = scanlines.intersection(features.geometry)

        intersection_table = []
        for f_index, inter in enumerate(all_intersections):
            for ml in inter.geoms:
                # f_index, y, x0, x1, (space for row, col1,col2) 
                coords = np.asarray(ml.coords)
                y_ = coords[0][1]
                x0 = coords[0][0]
                x1 = coords[-1][0]
                intersection_table.append((
                    f_index, y_, x0, x1
                ))

        intersection_table = np.array(intersection_table)
        inter_x0s = intersection_table[:, 2]
        inter_x1s = intersection_table[:, 3]
        inter_ys = intersection_table[:, 1]
        f_index = intersection_table[:, 0]

        rows, col0s = xy_to_rowcol(inter_x0s, inter_ys)
        _, col1s = xy_to_rowcol(inter_x1s, inter_ys)
        reading_table = np.stack([
            rows, col0s, col1s, f_index
        ], axis=1).astype(int)

        # sort by row
        reading_table = reading_table[np.argsort(reading_table[:, 0])]
        rows, row_starts = np.unique(reading_table[:, 0], return_index=True)

        pixel_values_per_feature = [[] for _ in range(len(all_intersections))]

        for i, row in enumerate(rows):
            start = row_starts[i]
            end = row_starts[i+1] if i+1 < len(row_starts) else len(reading_table)
            reading_line = reading_table[start:end]
            
            min_col = np.min(reading_line[:, 1])
            max_col = np.max(reading_line[:, 2])
            reading_window = rio.windows.Window(
                col_off=min_col,
                row_off=row,
                width=max_col - min_col,
                height=1
            )
            # Does not handle nodata
            data = raster.read(1, window=reading_window, masked=False)[0]
            for j, col0, col1, f_index in reading_line:
                c0 = col0 - min_col
                c1 = col1 - min_col
                pixel_values = data[c0:c1]
                
                # Debugging code
                # mark the pixels in the global mask
                # row_in_mask = row - row_start
                # col0_in_mask = col0 - col_start
                # col1_in_mask = col1 - col_start
                # global_mask[row_in_mask, col0_in_mask:col1_in_mask] = f_index + 1
                # End Debugging code

                if len(pixel_values) > 0:
                    pixel_values_per_feature[f_index].append(pixel_values)
        

        # combine the results
        results_per_feature = []
        for i in range(len(pixel_values_per_feature)):
            feature_data = np.concatenate(pixel_values_per_feature[i])
            # get the stats
            r = self._compute_stats_from_array(feature_data)
            results_per_feature.append(r)

        self.results = results_per_feature

        # Debugging code
        # from raster_methods import Masking
        # m = Masking()
        # ref_mask = np.zeros_like(global_mask, dtype=int)
        # for i in range(len(features.geometry)):
        #     # get all results for this feature
        #     feature = features.geometry[i]
        #     fmask = m.mask(feature, raster, window)
        #     # add a 0 row at the end of fmask
        #     fmask = np.insert(fmask, 0, 0, axis=0)
        #     ref_mask[fmask] = i+1
        # import matplotlib.pyplot as plt
        # diff_mask = global_mask - ref_mask
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(diff_mask, cmap='inferno')
        # ax.set_title('Difference Mask')
        # plt.savefig('diff_mask.png')
        # plt.close()
        # end of debugging code    
        
    
    def _run(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        self._precomputations2(features, raster)
        return self.results

    # def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):

    #     # just fetch the results
    #     return self.results



def test():
    from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH
    from reference import reference_method
    ag = Scanline2()
    vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
    raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR_resampled_x4.tif'

    # test_raster()
    # raster_layer_file = f'{RASTER_DATA_PATH}/test.tif'
    r1 = ag(raster_layer_file, vector_layer_file, ['count'])

    rlow, rhigh = reference_method(raster_layer_file, vector_layer_file, ['count'])

    for i in range(len(r1)):
        r_c = r1[i]['count']
        r_l = rlow[i]['count']
        r_h = rhigh[i]['count']
        err1 = abs(r_c - r_l) / r_l
        err2 = abs(r_c - r_h) / r_h
        print(f"Error {i}: {err1:.2%}")



    print('done')



# test()













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

    def _get_params(self):
        return {
            "max_depth": self.max_depth
        }
    
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

        # print(f"Total boxes: {n_total_boxes}")
        # print(f"Width: {width}, Height: {height} in coordinates")
        # print(f"Minx: {x_min}, Miny: {y_min} in coordinates")
        # print(f"Maxx: {x_max}, Maxy: {y_max} in coordinates")
        # print(f"Raster size: {raster.width} x {raster.height}")

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

            # print(f"Level {level}: {n_boxes} boxes")
            # print(f"Divisions: {divisions}")
            # print(f"Len boxes: {len(boxes)}")
            # print(f"max boxes: {max(boxes[:, 2])}, {max(boxes[:, 3])}")
            # print(f"min boxes: {min(boxes[:, 0])}, {min(boxes[:, 1])}")
            # print(f"First box: {boxes[0]}")

            ids = (np.arange(n_boxes) + box_index).astype('uint8')
            box_index += n_boxes
            parent_ids = (np.arange(n_boxes // 4) + box_index).astype('uint8').repeat(4)
            if level == 0:
                parent_ids = np.array([np.nan])

            shp_boxes1 = box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
            shp_boxes = [box(x0, y0, x1, y1) for x0, y0, x1, y1 in boxes]
            # print(shp_boxes[0])
            # print(shp_boxes1[0])
            # print(boxes[0])
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
                # print(f"Data shape: {data.shape}")

                masking_method = Masking()
                aggregates = masking_method(raster, vector_layer, self.stats)
                aggregates = np.array(aggregates)
                # print(f"Aggregates count: {aggregates[0]['count']}")
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
                # print(f"Fully conained node")

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
                    # print(f"Not contained leaf")
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
    plt.show()
    print('stop')
    # add colorbar to the first two plots
    
    # plot difference between the two masks
    # mask - correct_mask
    # print('done')

    


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