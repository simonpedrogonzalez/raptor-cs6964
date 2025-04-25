from experiment import Experiment
from constants import VECTOR_DATA_PATH, RASTER_DATA_PATH
from reference import reference_method
from raptor_methods import AggQuadTree, Scanline
from raster_methods import Masking, Clipping
from vector_methods import NaivePointInPolygon, QSplit
import time

vector_layer_file = f'{VECTOR_DATA_PATH}/cb_2018_us_state_20m_filtered.shp'
raster_layer_file = f'{RASTER_DATA_PATH}/US_MSR.tif'

# change the shape file crs from 4269 to 3857
# and save it to the same file
# vector = gpd.read_file(vector_layer_file)
# vector.to_crs(epsg=3857, inplace=True)
# vector.to_file(vector_layer_file)

exp1 = Experiment(
    raster_path=raster_layer_file,
    vector_path=vector_layer_file,
    func=NaivePointInPolygon(),
    reps=5,
    stats=['mean'],
    check_results=False,
)

exp2 = Experiment(
    raster_path=raster_layer_file,
    vector_path=vector_layer_file,
    func=QSplit(),
    reps=5,
    stats=['mean'],
    check_results=False,
)

exp3 = Experiment(
    raster_path=raster_layer_file,
    vector_path=vector_layer_file,
    func=Masking(),
    reps=5,
    stats=['mean'],
    check_results=False,
)

exp4 = Experiment(
    raster_path=raster_layer_file,
    vector_path=vector_layer_file,
    func=Clipping(),
    reps=5,
    stats=['mean'],
    check_results=False,
)
exp5 = Experiment(
    raster_path=raster_layer_file,
    vector_path=vector_layer_file,
    func=Scanline(),
    reps=5,
    stats=['mean'],
    check_results=False,
)
exp6 = Experiment(
    raster_path=raster_layer_file,
    vector_path=vector_layer_file,
    func=AggQuadTree(),
    reps=5,
    stats=['mean'],
    check_results=False,
)

if __name__ == "__main__":
    t0 = time.time()
    print("Starting all experiments")
    exp1.run()
    exp2.run()
    exp3.run()
    exp4.run()
    exp5.run()
    exp6.run()
    t1 = time.time()
    dt = t1 - t0
    dt_in_minutes = dt / 60
    dt_in_hours = dt / 3600
    if dt_in_hours > 1:
        remaining_minutes = dt_in_minutes % 60
        remaining_seconds = dt % 60
        print(f"All experiments finished in {dt_in_hours:.2f}:{remaining_minutes:.2f}:{remaining_seconds:.2f} hours")
    elif dt_in_minutes > 1:
        remaining_seconds = dt % 60
        print(f"All experiments finished in {dt_in_minutes:.2f}:{remaining_seconds:.2f} minutes")
    else:
        print(f"All experiments finished in {dt:.2f} seconds")
    print("All experiments finished")