from experiment import Experiment
from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH
from reference import reference_method
from raptor_methods import AggQuadTree, Scanline, Scanline2
from raster_methods import Masking, Clipping
from vector_methods import NaivePointInPolygon, QSplit
import time

vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR_upsampled_5.tif'

# change the shape file crs from 4269 to 3857
# and save it to the same file
# vector = gpd.read_file(vector_layer_file)
# vector.to_crs(epsg=3857, inplace=True)
# vector.to_file(vector_layer_file)

exps = [
    Experiment(
        raster_path=raster_layer_file,
        vector_path=vector_layer_file,
        func=func,
        reps=5,
    )
    for func in [
        Masking(),
        # Scanline(),
        # AggQuadTree(),
        Scanline2(),
    ]
]

if __name__ == "__main__":
    from experiment_aggregator import ExperimentAggregator
    exp_agg = ExperimentAggregator(exps)
    exp_agg.run()