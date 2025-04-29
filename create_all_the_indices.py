from raptor_methods import AggQuadTree
from constants import US_states, US_MSR_resampled_x2,\
    US_MSR_resampled_x4, US_MSR, US_MSR_upsampled_2,\
        US_MSR_upsampled_4
import tqdm

# Run  on the US states with every raster file, using depths 5, 7, 10

rasters = [
    US_MSR_resampled_x2,
    US_MSR_resampled_x4,
    US_MSR,
    US_MSR_upsampled_2,
    US_MSR_upsampled_4
]

depths = [
    5, 6, 7, 8
]

for raster in rasters:
    for depth in depths:
        print(f"Running  on {raster} with depth {depth}...")
        agg = AggQuadTree(max_depth=depth)
        agg(raster, US_states, ["count", "mean", "sum"])
        print(f"Finished running  on {raster} with depth {depth}.")
