import rasterio
from rasterio.enums import Resampling
from constants import DATA_PATH, RASTER_DATA_PATH

def upsample_tiff(input_path, output_path, scale_factor):
    with rasterio.open(input_path) as src:
        # Calculate the new shape
        new_height = src.height * scale_factor
        new_width = src.width * scale_factor

        # Update the transform (each pixel covers less area now)
        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height
        )

        # Read and resample the data
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear  # Or nearest, cubic, etc.
        )

        # Update metadata
        profile = src.profile
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })

        # Write the oversampled raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

file = f'{RASTER_DATA_PATH}/US_MSR.tif'
scale_factor = 4  # Change this to your desired scale factor
output_file = f'{RASTER_DATA_PATH}/US_MSR_upsampled_{scale_factor}.tif'
print(f"Upsampling {file} to {output_file} with scale factor {scale_factor}...")
upsample_tiff(file, output_file, scale_factor)
print("Upsampling completed.")