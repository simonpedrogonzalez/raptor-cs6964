# raptor-cs6964

Raster, Vector and Raptor Based approaches for Zonal Statistics calculations on GIS data.

## Environment setup

### Using uv manager

1. Clone the repo
2. uv installation https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
2. uv python version installation: `uv python install 3.12`
3. venv creation and dependency installation: cd to the project directory and run `uv sync`

### Using other tools

If you want use conda, pipenv, pip or whatever, I exported a requirements.txt file with the dependencies, which you can use to install them.

## Running the code

From console: activate the environment with `source .venv/bin/activate` and run the script with `python src/my_experiment.py`.

If using vscode and uv, remember to setup the python interpreter used for your debugger: open command pallete (ctrl+shift+p) and search for "Python: Select Interpreter", then select the one in the .venv directory, which should be something like `Python 3.12.8 ('.venv')`.

Then just "Run and Debug" as usual.

## Troubleshooting

0. Send an email to `u1528314@umail.utah.edu`, I'll be happy to help.

1. If not present in the repo, create directories:
`data/raster`
`data/vector`
`data/indices`
`results`

2. If you get an error when running AggQuadTree, check if the index files for the raster data that you are using are already present. If the index files were produced with a diferent "stats" configuration, they might not be compatible with your current run. Hence, delete the index files and the method will generate them again.

3. If you are receiving the zip, all data and indices should already be present.

## Files:

Method implementation: `vector_methods.py`, `raster_methods.py`, `raptor_methods.py`, `zonal_stats.py`, `node.py`

Benchmarking framework implementation: `experiment.py`, `experiment_aggregator.py`

Preprocessing scripts: `preprocess.py`, `oversample_raster.py`

Replicating reported experiments: `exp_all_algorithms_in_default_raster.py`, `exp_fast_alg_only.py`, `exp_fast_alg_only_raster_scaling.py`, `exp_fast_alg_only_vector_scaling.py`, `exp_tune_agg_qtree.py`.

Kafka Files: `kafka/`