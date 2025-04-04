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
from shapely import Polygon, Point, LineString
from typing import List, Tuple, Dict
from rtree import index

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
    def __init__(self, ident, level, bounds, children):
        self.id = ident
        self.level = level
        self.bounds = bounds
        self.children = children
        self.stats = None

class AggQuadTree(ZonalStatMethod):

    
    __name__ = "AggQuadTree"

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        super().__init__()

    # def _build_agg_quadtree(self, raster: rio.DatasetReader, node: Node, depth: int, 


    def _precomputations(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader):
        
        width = raster.width
        height = raster.height

        width = 10
        height = 10

        boxes_per_level = {}
        x_min, y_min = 0, 0 # this could be set to coords instead
        x_max, y_max = width, height # this could be set to coords instead
        n_total_boxes = sum([4 ** i for i in range(self.max_depth + 1)])
        box_index = 0
        idx = index.Index()

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
            ids = np.arange(box_index, box_index + len(boxes))
            box_index += len(boxes)
            parent_ids = np.floor(ids / 4).astype(int) + box_index
            
            for i in range(len(boxes)):
                idx.insert(ids[i], boxes[i], parent_ids[i])
            
            boxes_per_level[level] = [
                {"id": ids[i], "bounds": boxes[i], "parent_id": parent_ids[i]}
                for i in range(len(boxes))
            ]

        print(boxes_per_level)
        print('done')


            
            # box_ids = 
            

        # # given that each node has 4 children, the leaf nodes will be 4^max_depth
        # n_leaf_nodes = 4 ** max_depth
        # # compute all the boxes for the leaf nodes
        # boxes = []
        # for i in range(n_leaf_nodes):


        # idx = index.Index()



        # Create a quadtree index
        # root = Node(0, 0, (0, 0, 10, 10), [])
        # idx = index.Index()

        # idx.insert(root.id, root.bounds, root)

        # # Split the root node into quadrants
        # node1 = Node(1, 1, (0, 0, 5, 5), [])
        # node2 = Node(2, 1, (5, 0, 10, 5), [])
        # node3 = Node(3, 1, (0, 5, 5, 10), [])
        # node4 = Node(4, 1, (5, 5, 10, 10), [])

        # idx.insert(node1.id, node1.bounds, node1)
        # idx.insert(node2.id, node2.bounds, node2)
        # idx.insert(node3.id, node3.bounds, node3)
        # idx.insert(node4.id, node4.bounds, node4)

        # node6 = Node(6, 2, (0, 0, 2.5, 2.5), [])
        # node7 = Node(7, 2, (2.5, 0, 5, 2.5), [])
        # node8 = Node(8, 2, (0, 2.5, 2.5, 5), [])
        # node9 = Node(9, 2, (2.5, 2.5, 5, 5), [])

        # idx.insert(node6.id, node6.bounds, node6)
        # idx.insert(node7.id, node7.bounds, node7)
        # idx.insert(node8.id, node8.bounds, node8)
        # idx.insert(node9.id, node9.bounds, node9)

        # node10 = Node(10, 2, (5, 0, 7.5, 2.5), [])
        # node11 = Node(11, 2, (7.5, 0, 10, 2.5), [])
        # node12 = Node(12, 2, (5, 2.5, 7.5, 5), [])
        # node13 = Node(13, 2, (7.5, 2.5, 10, 5), [])

        # idx.insert(node10.id, node10.bounds, node10)
        # idx.insert(node11.id, node11.bounds, node11)
        # idx.insert(node12.id, node12.bounds, node12)
        # idx.insert(node13.id, node13.bounds, node13)



        # query = (3.25, 1.25, 6.25, 3.25) # should return 0, 1, 2, 7, 9, 10, 12.

        # result = list(idx.intersection(query))
        # print(result)

        print('done')

    
def test():

    from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH

    ag = AggQuadTree(max_depth=3)
    vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
    raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR.tif'

    ag(raster_layer_file, vector_layer_file, ['count', 'mean'])

test()