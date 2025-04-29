from kafka import KafkaProducer
import geopandas as gpd
from shapely.geometry import Polygon, box
import pickle
from io import BytesIO
import time
import rasterio
import numpy as np
import os

# Function to serialize polygon data
def serialize(polygon_data):
    buffer = BytesIO()
    pickle.dump(polygon_data, buffer)
    return buffer.getvalue()

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: serialize(v)
)

# Path to your TIFF files to extract extents
tiff_directory = "/Users/devagopalam/desktop/raptor-cs6964-main/dataset"

# Function to create a grid of polygons over a raster extent
def create_polygon_grid(raster_path, grid_size=10):
    """
    Create a grid of polygons over a raster's extent.
    grid_size: number of cells in each dimension (resulting in grid_size^2 total polygons)
    """
    polygons = []
    
    with rasterio.open(raster_path) as src:
        # Get the bounding box of the raster
        left, bottom, right, top = src.bounds
        
        # Calculate cell dimensions
        width = (right - left) / grid_size
        height = (top - bottom) / grid_size
        
        # Create grid cells
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate cell coordinates
                cell_left = left + col * width
                cell_bottom = bottom + row * height
                cell_right = cell_left + width
                cell_top = cell_bottom + height
                
                # Create polygon for this cell
                poly = box(cell_left, cell_bottom, cell_right, cell_top)
                
                # Add to list with an ID
                polygons.append({
                    'geometry': poly,
                    'properties': {
                        'id': f'grid_r{row}_c{col}',
                        'row': row,
                        'col': col
                    },
                    'id': f'grid_r{row}_c{col}'
                })
    
    return polygons

# Find the first TIFF file to use as reference for creating polygons
tiff_file = None
for filename in os.listdir(tiff_directory):
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        tiff_file = os.path.join(tiff_directory, filename)
        break

if tiff_file is None:
    print("No TIFF files found in the specified directory!")
    exit(1)

print(f"Creating synthetic polygons based on the extent of: {tiff_file}")

# Create grid polygons
grid_polygons = create_polygon_grid(tiff_file, grid_size=5)  # 5x5 grid = 25 polygons

print(f"Generated {len(grid_polygons)} synthetic polygons")

# Send polygons to Kafka
for polygon in grid_polygons:
    # Send polygon to Kafka
    producer.send("raw-polygons", value=polygon)
    
    # Small delay to avoid overwhelming the system
    time.sleep(0.1)

# Ensure all messages are sent
producer.flush()

print("All synthetic polygons have been sent to Kafka")