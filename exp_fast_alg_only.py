from experiment import Experiment
from constants import US_states, US_MSR_resampled_x4, US_MSR, US_MSR_upsampled_4
from raster_methods import RasterStatsMasking, Masking
from raptor_methods import AggQuadTree, Scanline
from experiment_aggregator import ExperimentAggregator


exps = [
    Experiment(
        raster_path=US_MSR_upsampled_4,
        vector_path=US_states,
        func=func,
        reps=10,
        stats=["count"],
    )
    for func in [
        RasterStatsMasking(),
        Masking(),
        Scanline(),
        AggQuadTree(max_depth=5),
    ]
]

if __name__ == "__main__":
    ExperimentAggregator(exps).run()
