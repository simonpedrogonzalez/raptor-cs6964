import numpy as np
import rasterio
from shapely.geometry import Polygon, Point
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class RasterInfo:
    """Class to store raster metadata"""
    transform: rasterio.Affine
    bounds: rasterio.coords.BoundingBox
    height: int
    width: int
    nodata: float

class RaptorTIFF:
    def __init__(self, tif_path: str):
        """
        Initialize Raptor analysis with a GeoTIFF file.

        Args:
            tif_path: Path to the GeoTIFF file
        """
        self.tif_path = tif_path
        self.raster_info = None
        self.dataset = None
        self._load_raster()

    def _load_raster(self):
        """Load and store raster metadata"""
        with rasterio.open(self.tif_path) as src:
            self.raster_info = RasterInfo(
                transform=src.transform,
                bounds=src.bounds,
                height=src.height,
                width=src.width,
                nodata=src.nodata
            )

    def _pixel_to_coord(self, row: int, col: int) -> Tuple[float, float]:
        """Convert pixel indices to coordinates"""
        x, y = rasterio.transform.xy(self.raster_info.transform, row, col)
        return (x, y)

    def _coord_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates to pixel indices"""
        row, col = rasterio.transform.rowcol(self.raster_info.transform, x, y)
        return (row, col)

    def compute_scanline_intersections(self, y: float, polygon: Polygon) -> List[float]:
        """
        Compute intersections of a horizontal scanline with polygon edges.

        Args:
            y: y-coordinate of the scanline
            polygon: Shapely Polygon object

        Returns:
            Sorted list of x-coordinates where scanline intersects polygon
        """
        # Create a horizontal line at y
        bounds = polygon.bounds
        line_start = Point(bounds[0] - 1, y)
        line_end = Point(bounds[2] + 1, y)

        # Create a horizontal line and get intersections
        scanline = LineString([line_start, line_end])
        if not scanline.intersects(polygon):
            return []

        intersection = scanline.intersection(polygon)

        # Handle different types of intersections
        if intersection.geom_type == 'Point':
            return [intersection.x]
        elif intersection.geom_type == 'MultiPoint':
            return sorted([p.x for p in intersection])
        elif intersection.geom_type == 'LineString':
            return sorted([intersection.coords[0][0], intersection.coords[-1][0]])
        elif intersection.geom_type == 'MultiLineString':
            xs = []
            for line in intersection:
                xs.extend([line.coords[0][0], line.coords[-1][0]])
            return sorted(xs)
        return []

    def process_scanline(self, y: float, intersections: List[float], band_data: np.ndarray) -> Dict[str, float]:
        """
        Process pixels between intersection pairs on a scanline.

        Args:
            y: y-coordinate of the scanline
            intersections: Sorted list of x-intersections
            band_data: Raster band data

        Returns:
            Dictionary containing sum and count of processed pixels
        """
        sum_values = 0.0
        count = 0

        # Process pairs of intersections
        for i in range(0, len(intersections), 2):
            if i + 1 >= len(intersections):
                break

            # Convert start and end points to pixel coordinates
            start_row, start_col = self._coord_to_pixel(intersections[i], y)
            end_row, end_col = self._coord_to_pixel(intersections[i + 1], y)

            # Ensure we're within raster bounds
            start_col = max(0, start_col)
            end_col = min(self.raster_info.width, end_col + 1)

            if 0 <= start_row < self.raster_info.height:
                # Process pixels between intersections
                for col in range(start_col, end_col):
                    value = band_data[start_row, col]
                    if value != self.raster_info.nodata and not np.isnan(value):
                        sum_values += value
                        count += 1

        return {"sum": sum_values, "count": count}

    def compute_zonal_statistics(self, polygon: Polygon, band_idx: int = 1) -> Dict[str, float]:
        """
        Compute zonal statistics for a polygon using the scanline method.

        Args:
            polygon: Shapely Polygon object
            band_idx: Index of the raster band to process (1-based)

        Returns:
            Dictionary containing calculated statistics
        """
        total_sum = 0.0
        total_count = 0

        # Get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds

        with rasterio.open(self.tif_path) as src:
            # Read the specified band
            band_data = src.read(band_idx)

            # Process each scanline
            y = miny
            while y <= maxy:
                # Get intersections for this scanline
                intersections = self.compute_scanline_intersections(y, polygon)

                if len(intersections) >= 2:
                    # Process the scanline
                    result = self.process_scanline(y, intersections, band_data)
                    total_sum += result["sum"]
                    total_count += result["count"]

                # Move to next pixel row in y direction
                _, row = self._coord_to_pixel(minx, y)
                y_next = self._pixel_to_coord(row + 1, 0)[1]
                y = y_next

        # Calculate final statistics
        stats = {
            "count": total_count,
            "sum": total_sum,
            "mean": total_sum / total_count if total_count > 0 else np.nan,
            "nodata_value": self.raster_info.nodata
        }

        return stats

# Example usage
if __name__ == "__main__":
    import shapely.geometry

    # Example with sample data
    tif_file = "src/data/US_MSR.tif"

    # Create a sample polygon (replace with your actual polygon coordinates)
    polygon_coords = [
        (-120.5, 37.5), (-120.0, 37.5),
        (-120.0, 38.0), (-120.5, 38.0),
        (-120.5, 37.5)
    ]
    polygon = shapely.geometry.Polygon(polygon_coords)

    try:
        # Initialize Raptor analysis
        raptor = RaptorTIFF(tif_file)

        # Compute statistics
        results = raptor.compute_zonal_statistics(polygon)

        print("Zonal Statistics Results:")
        print(f"Count of pixels: {results['count']}")
        print(f"Sum of values: {results['sum']:.2f}")
        print(f"Mean value: {results['mean']:.2f}")

    except Exception as e:
        print(f"Error processing data: {str(e)}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as PlotPolygon
from shapely.geometry import Polygon, LineString
import rasterio
from rasterio.features import geometry_mask

class RaptorVisualizer:
    def __init__(self, raster_path):
        self.raster = rasterio.open(raster_path)
        self.transform = self.raster.transform

    def visualize_scanline_method(self, polygon, output_path):
        """Visualize scanline method similar to Figure 4(a) in the paper."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set up grid
        grid_size = 10
        for i in range(grid_size):
            for j in range(grid_size):
                ax.add_patch(Rectangle((i, j), 1, 1, fill=False, color='gray', alpha=0.3))

        # Plot polygon
        x, y = polygon.exterior.xy
        poly_patch = PlotPolygon(np.column_stack([x, y]), facecolor='none',
                               edgecolor='black', linewidth=2)
        ax.add_patch(poly_patch)

        # Generate and plot scanlines
        minx, miny, maxx, maxy = polygon.bounds
        y_coords = np.linspace(miny, maxy, 20)

        for y in y_coords:
            # Create horizontal scanline
            line = LineString([(minx-1, y), (maxx+1, y)])
            if polygon.intersects(line):
                intersections = polygon.intersection(line)
                if hasattr(intersections, 'geoms'):
                    for part in intersections.geoms:
                        x, y = part.xy
                        ax.plot(x, y, 'r--', linewidth=1)
                else:
                    x, y = intersections.xy
                    ax.plot(x, y, 'r--', linewidth=1)

        # Add arrows on right side
        for y in np.linspace(0, grid_size-1, 8):
            ax.arrow(grid_size, y, 1, 0, head_width=0.2, head_length=0.2,
                    fc='black', ec='black')

        ax.set_xlim(-1, grid_size+2)
        ax.set_ylim(-1, grid_size+1)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.savefig(f"{output_path}_scanline.png", bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_aggquadtree(self, polygon, output_path):
        """Visualize aggregate quadtree method similar to Figure 4(b) in the paper."""
        fig, ax = plt.subplots(figsize=(10, 10))

        def draw_quadtree(bounds, depth=0, max_depth=3):
            x, y, w, h = bounds

            # Create rectangle for current quad
            rect = Rectangle((x, y), w, h, fill=False, color='gray', alpha=0.3)
            ax.add_patch(rect)

            # If intersects with polygon and not at max depth, subdivide
            rect_poly = Polygon([(x,y), (x+w,y), (x+w,y+h), (x,y+h)])
            if depth < max_depth and polygon.intersects(rect_poly):
                # Subdivide into four quadrants
                w2, h2 = w/2, h/2
                draw_quadtree((x, y, w2, h2), depth+1)         # SW
                draw_quadtree((x+w2, y, w2, h2), depth+1)      # SE
                draw_quadtree((x, y+h2, w2, h2), depth+1)      # NW
                draw_quadtree((x+w2, y+h2, w2, h2), depth+1)   # NE

                # Add shading for cells that intersect with polygon
                if polygon.intersects(rect_poly):
                    overlap_area = polygon.intersection(rect_poly).area
                    if overlap_area > 0:
                        rect = Rectangle((x, y), w, h,
                                      facecolor='gray', alpha=overlap_area/rect_poly.area)
                        ax.add_patch(rect)

        # Draw initial quadtree
        draw_quadtree((0, 0, 10, 10))

        # Plot polygon
        x, y = polygon.exterior.xy
        poly_patch = PlotPolygon(np.column_stack([x, y]), facecolor='none',
                               edgecolor='black', linewidth=2)
        ax.add_patch(poly_patch)

        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.savefig(f"{output_path}_aggquadtree.png", bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # Example usage
    raster_path = "src/data/US_MSR.tif"
    visualizer = RaptorVisualizer(raster_path)

    # Create example polygon (similar to California shape in the paper)
    polygon = Polygon([
        (3, 2), (4, 2), (5, 3), (6, 4), (7, 6),
        (6, 7), (5, 8), (4, 7), (3, 6), (3, 2)
    ])

    # Generate both visualizations
    visualizer.visualize_scanline_method(polygon, "raptor_viz")
    visualizer.visualize_aggquadtree(polygon, "raptor_viz")

if __name__ == "__main__":
    main()

import rasterio
import numpy as np
from rasterio.features import geometry_mask
from shapely.geometry import Polygon
import numpy.ma as ma

class ZonalStats:
    def __init__(self, raster_path):
        """Initialize with path to raster file."""
        self.raster = rasterio.open(raster_path)
        self.transform = self.raster.transform

    def compute_stats(self, polygon):
        """Compute zonal statistics for given polygon."""
        # Create mask for the polygon
        mask = geometry_mask([polygon],
                           out_shape=(self.raster.height, self.raster.width),
                           transform=self.raster.transform,
                           invert=True)

        # Read masked data
        data = self.raster.read(1, masked=True)
        masked_data = ma.masked_array(data, ~mask)

        # Calculate basic statistics
        stats = {
            'mean': float(masked_data.mean()),
            'std': float(masked_data.std()),
            'min': float(masked_data.min()),
            'max': float(masked_data.max()),
            'count': int(masked_data.count()),
            'sum': float(masked_data.sum()),
            'median': float(np.ma.median(masked_data)),
            '25th_percentile': float(np.percentile(masked_data.compressed(), 25)),
            '75th_percentile': float(np.percentile(masked_data.compressed(), 75))
        }

        return stats

def main():
    # Example usage
    raster_path = "src/data/US_MSR.tif"  # Replace with your .tif file path
    calculator = ZonalStats(raster_path)

    # Get raster bounds
    bounds = calculator.raster.bounds
    min_x, min_y, max_x, max_y = bounds

    # Create a test polygon (adjust coordinates based on your raster extent)
    width = (max_x - min_x) * 0.3
    height = (max_y - min_y) * 0.3
    offset_x = (max_x - min_x) * 0.1
    offset_y = (max_y - min_y) * 0.1

    polygon = Polygon([
        (min_x + offset_x, min_y + offset_y),
        (min_x + offset_x + width, min_y + offset_y),
        (min_x + offset_x + width, min_y + offset_y + height),
        (min_x + offset_x, min_y + offset_y + height),
        (min_x + offset_x, min_y + offset_y)
    ])

    # Calculate statistics
    stats = calculator.compute_stats(polygon)

    # Print results
    print("\nZonal Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()