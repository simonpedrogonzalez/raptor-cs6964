from experiment import Experiment
from constants import US_states, US_MSR_resampled_x4, US_MSR, US_MSR_upsampled_4
from raster_methods import RasterStatsMasking, Masking, Clipping
from vector_methods import NaivePointInPolygon, QSplit
from raptor_methods import AggQuadTree, Scanline
from experiment_aggregator import ExperimentAggregator

# change the shape file crs from 4269 to 3857
# and save it to the same file
# vector = gpd.read_file(vector_layer_file)
# vector.to_crs(epsg=3857, inplace=True)
# vector.to_file(vector_layer_file)

exps = [
    Experiment(
        raster_path=US_MSR_upsampled_4,
        vector_path=US_states,
        func=func,
        reps=10,
        stats=["count"],
    )
    for func in [
        Masking(),
        AggQuadTree(max_depth=5),
        AggQuadTree(max_depth=6),
        AggQuadTree(max_depth=7),
        AggQuadTree(max_depth=8),
    ]
]

if __name__ == "__main__":
    ExperimentAggregator(exps).run()
