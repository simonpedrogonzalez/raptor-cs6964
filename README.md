# raptor-cs6964

Raster, Vector and Raptor Based approaches for aggregate calculations on GIS data.

## Recommendations for the project

1. I like using gitkraken for managing the repository, you should have access to the pro version if you requested the student github pack with you university email.

2. I like vscode as IDE.

## Environment setup

### Using uv manager

uv is my favorite because:
- it covers all needs (managing python versions, virtual envs and dependency resolution)
- it's ridiculous how fast it is
- the `.venv` is created by default in the project directory:
    - allows inspecting dependencies code easily (useful for checking how a library implements a zonal statistic method, for example)
    - keeps each project's dependencies isolated explicitly

1. uv installation https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
2. uv python version installation: `uv python install 3.12` (recommended because is about 60% faster than 3.10 or previous versions)
3. venv creation and dependency installation: cd to the project directory and run `uv sync`

Other things you may want to do with uv:
- `uv python list` to see all installed python versions
- `uv add <package>` to install a package in the current venv

### Using other tools

If you wanna use conda, pipenv, pip or whatever, I exported a requirements.txt file with the dependencies, which you can use to install them.

## Running the code

If using vscode and uv, remember to setup the python interpreter used for your debugger: open command pallete (ctrl+shift+p) and search for "Python: Select Interpreter", then select the one in the .venv directory, which should be something like `Python 3.12.8 ('.venv')`.

Then just "Run and Debug" as usual.

## Useful stuff
- Install QGIS in linux: http://test.qgis.org/html/en/site/forusers/alldownloads.html#debian-ubuntu

## Dataset sources

Raster data sources:
- https://cds.climate.copernicus.eu/datasets

Vector data sources:
- https://www.gadm.org
- https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
