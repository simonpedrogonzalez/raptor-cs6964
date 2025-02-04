# Define these variables
export PROJ_LIB="/Users/u1528314/Applications/QGIS-LTR.app/Contents/Resources/proj"
export GDAL_DATA="/Users/u1528314/Applications/QGIS-LTR.app/Contents/Resources/gdal"
# Install debugpy in QGIS python3
/Users/u1528314/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3 -m pip install debugpy
# then set the interpreter in vscode to the QGIS python3 interpreter
# then run the debug configuration "Attach to QGIS" in vscode