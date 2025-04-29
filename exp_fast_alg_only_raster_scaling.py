from experiment import Experiment
from constants import US_states, US_MSR_resampled_x4, US_MSR_resampled_x2, \
    US_MSR, US_MSR_upsampled_4, US_MSR_upsampled_2
from raster_methods import RasterStatsMasking, Masking
from raptor_methods import AggQuadTree, Scanline
from experiment_aggregator import ExperimentAggregator

# change the shape file crs from 4269 to 3857
# and save it to the same file
# vector = gpd.read_file(vector_layer_file)
# vector.to_crs(epsg=3857, inplace=True)
# vector.to_file(vector_layer_file)

rasters = [
    US_MSR,
    US_MSR_resampled_x2,
    US_MSR_resampled_x4,
    US_MSR_upsampled_2,
    US_MSR_upsampled_4,
]

funcs = [
    RasterStatsMasking(),
    Masking(),
    Scanline(),
    AggQuadTree(max_depth=5),
]

exps = []

for raster in rasters:
    for func in funcs:
        exps.append(
            Experiment(
                raster_path=raster,
                vector_path=US_states,
                func=func,
                reps=10,
                stats=["count"],
            )
        )

if __name__ == "__main__":
    ExperimentAggregator(exps).run()
