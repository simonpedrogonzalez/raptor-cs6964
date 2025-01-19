'''This should be more or less what to copypaste in QGIS python console, change base path to your own.
I think QGIS uses the masking method, but we should check that.
'''

from qgis.analysis import QgsZonalStatistics

base = '/home/simon/repos/mdml/raptor-cs6964/src/data'

vectorLayer = QgsVectorLayer(f'{base}/ne_10m_admin_0_countries.shp', 'vector_layer', "ogr")
assert vectorLayer.isValid()

rasterLayer = QgsRasterLayer(f'{base}/ECMWF_utci_20230101_v1.1_con.nc', 'raster_layer')
assert rasterLayer.isValid()

zoneStat = QgsZonalStatistics(vectorLayer, rasterLayer, 'pre-', 1, QgsZonalStatistics.Mean)
zoneStat.calculateStatistics(None)