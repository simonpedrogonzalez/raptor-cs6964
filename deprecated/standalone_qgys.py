import debugpy

# Start debug server and wait for VSCode to attach
debugpy.listen(("localhost", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached! Running script...")

from qgis.core import *
from qgis.analysis import QgsZonalStatistics

QgsApplication.setPrefixPath("/Users/u1528314/Applications/QGIS-LTR.app", True)
qgs = QgsApplication([], False)
qgs.initQgis()

# CODE HERE


base = 'src/data'

vectorLayer = QgsVectorLayer(f'{base}/ne_10m_admin_0_countries.shp', 'vector_layer', "ogr")
assert vectorLayer.isValid()

rasterLayer = QgsRasterLayer(f'{base}/ECMWF_utci_20230101_v1.1_con.nc', 'raster_layer')
assert rasterLayer.isValid()

zoneStat = QgsZonalStatistics(vectorLayer, rasterLayer, 'pre-', 1, QgsZonalStatistics.Mean)
zoneStat.calculateStatistics(None)

# Close the QGIS application
qgs.exitQgis()

