from experiment import Experiment
from experiment_aggregator import ExperimentAggregator
from constants import US_states, US_MSR_resampled_x2, US_MSR_resampled_x4, US_MSR
from raptor_methods import Scanline, AggQuadTree
from raster_methods import RasterStatsMasking

if __name__ == "__main__":
    methods = [ AggQuadTree(max_depth=5), Scanline(), RasterStatsMasking() ]
    rasters = [ US_MSR_resampled_x2 ]
    exps = [
        Experiment(
            raster_path=raster,
            vector_path=US_states,
            func=method,
            reps=2,
            stats=["count", "mean", "sum"],
            json_output=True, csv_output=True,
        )
        for method in methods
        for raster in rasters
    ]
    ExperimentAggregator(exps).run()

