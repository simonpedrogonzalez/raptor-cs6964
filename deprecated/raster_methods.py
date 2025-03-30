import numpy as np
import rasterio
from rasterio import features, mask
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import time

class RasterProcessor:
    def __init__(self, raster_path: str):
        """Initialize with just raster path since we'll create our own vector data"""
        self.raster_path = raster_path
        print(f"Initialized RasterProcessor with raster: {raster_path}")

    def check_raster(self):
        """Check if raster can be opened and print its properties"""
        try:
            with rasterio.open(self.raster_path) as src:
                print("\nRaster properties:")
                print(f"Shape: {src.shape}")
                print(f"CRS: {src.crs}")
                print(f"Bounds: {src.bounds}")
                return True
        except Exception as e:
            print(f"Error opening raster: {e}")
            return False

    def clip_raster(self):
        """Perform clipping operation"""
        try:
            print("\nStarting clipping operation...")
            with rasterio.open(self.raster_path) as src:
                print("Creating vector data...")
                # Create a simple polygon covering half the raster
                bounds = src.bounds
                polygon = box(bounds.left, bounds.bottom,
                           (bounds.left + bounds.right)/2, bounds.top)

                # Create GeoDataFrame with proper CRS
                gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=src.crs)
                print("Vector data created successfully")

                # Perform clipping
                print("Performing clipping...")
                clipped, _ = mask.mask(src, [polygon], crop=False, nodata=0)
                print("Clipping completed")
                return clipped[0]  # Return first band

        except Exception as e:
            print(f"Error in clipping operation: {e}")
            return None

    def mask_raster(self):
        """Perform masking operation"""
        try:
            print("\nStarting masking operation...")
            with rasterio.open(self.raster_path) as src:
                # Create same polygon as in clipping
                bounds = src.bounds
                polygon = box(bounds.left, bounds.bottom,
                           (bounds.left + bounds.right)/2, bounds.top)

                # Create mask layer
                mask_data = np.zeros(src.shape, dtype=np.uint8)

                # Rasterize vector to create mask
                print("Creating mask...")
                features.rasterize(
                    [(polygon, 1)],
                    out=mask_data,
                    transform=src.transform
                )
                print("Masking completed")
                return mask_data

        except Exception as e:
            print(f"Error in masking operation: {e}")
            return None

    def compute_statistics(self, clipped_data: np.ndarray, masked_data: np.ndarray):
        """Compute and print statistics"""
        try:
            print("\nComputing statistics...")

            # Statistics for clipped data
            valid_clipped = clipped_data[clipped_data > 0]

            print("\nClipping Method Statistics:")
            print(f"Mean value: {np.mean(valid_clipped):.2f}")
            print(f"Min value: {np.min(valid_clipped)}")
            print(f"Max value: {np.max(valid_clipped)}")
            print(f"Standard deviation: {np.std(valid_clipped):.2f}")
            print(f"Number of valid pixels: {len(valid_clipped)}")

            # Statistics for masked data
            masked_count = np.sum(masked_data == 1)

            print("\nMasking Method Statistics:")
            print(f"Number of masked pixels (1s): {masked_count}")
            print(f"Number of unmasked pixels (0s): {masked_data.size - masked_count}")
            print(f"Percentage of masked area: {(masked_count/masked_data.size)*100:.2f}%")

        except Exception as e:
            print(f"Error computing statistics: {e}")

    def create_paper_visualization(self, clip_result, mask_result, output_path):
        """Create visualization similar to paper's Figure 3"""
        try:
            print("\nCreating visualization...")

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Extract small 6x10 region near the boundary
            center_x = clip_result.shape[1] // 2
            center_y = clip_result.shape[0] // 2

            # Extract region
            region_x = slice(center_x-3, center_x+3)
            region_y = slice(center_y-5, center_y+5)

            clip_subset = clip_result[region_y, region_x]
            mask_subset = mask_result[region_y, region_x]

            # Normalize clipped values
            normalized = np.zeros_like(clip_subset, dtype=float)
            valid_mask = clip_subset > 0
            if valid_mask.any():
                min_val = clip_subset[valid_mask].min()
                max_val = clip_subset[valid_mask].max()
                normalized[valid_mask] = ((clip_subset[valid_mask] - min_val) /
                                       (max_val - min_val) * 140 + 60)

            # Plot clipping result
            ax1.imshow(normalized, cmap='gray')
            ax1.set_title('(a) Clipping')

            # Add value text for clipping
            for i in range(clip_subset.shape[0]):
                for j in range(clip_subset.shape[1]):
                    if clip_subset[i,j] > 0:
                        ax1.text(j, i, str(int(normalized[i,j])),
                               ha='center', va='center',
                               color='white' if normalized[i,j] > 127 else 'black')

            # Plot masking result
            ax2.imshow(mask_subset, cmap='binary')
            ax2.set_title('(b) Masking')

            # Add value text for masking
            for i in range(mask_subset.shape[0]):
                for j in range(mask_subset.shape[1]):
                    ax2.text(j, i, str(int(mask_subset[i,j])),
                            ha='center', va='center',
                            color='white' if mask_subset[i,j] == 1 else 'black')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved as {output_path}")

        except Exception as e:
            print(f"Error creating visualization: {e}")

# Run everything step by step
def main():
    # Initialize processor
    processor = RasterProcessor("src/data/US_MSR.tif")

    # Check if raster can be opened
    if not processor.check_raster():
        print("Failed to open raster file")
        return

    # Perform clipping
    t0 = time.time()
    clipped_data = processor.clip_raster()
    t1 = time.time()
    print(f"Clipping time: {t1-t0}")

    if clipped_data is None:
        print("Clipping operation failed")
        return

    # Perform masking
    t0 = time.time()
    masked_data = processor.mask_raster()
    t1 = time.time()
    print(f"Masking time: {t1-t0}")
    
    if masked_data is None:
        print("Masking operation failed")
        return

    # Compute statistics
    processor.compute_statistics(clipped_data, masked_data)

    # Create visualization
    processor.create_paper_visualization(clipped_data, masked_data, 'figure3_visualization.png')

if __name__ == "__main__":
    main()