# Raw data is downloaded and stored in rawData directory.
# From there, data is red and interpreted into numpy-arrays, which are stored in npyData directory.
# From there, data is fed into the model.


# Aktuelle Daten (Radar nur binär): https://opendata.dwd.de/weather/nwp/
# Historische Daten (aber keine Modellvorhersagen): ftp://ftp-cdc.dwd.de/pub/CDC/


urlRadolan = "ftp://ftp-cdc.dwd.de/pub/CDC/grids_germany/hourly/radolan/recent/asc/" # aktuelle radardaten in ascii format
urlCD2 = "https://opendata.dwd.de/weather/nwp/cosmo-d2/grib/" #aktuelle modelldaten in grib2 format
rawDataDir = "rawData"


def downloadFile(url, fileName, targetDir):
    pass


def downloadRadar(date):
    pass


def downloadModel(date):
    pass


def getModelData(fromTime, toTime, bbox, parameters):
    pass 


def getRadarData(fromTime, toTime, bbox, parameters):
    pass
