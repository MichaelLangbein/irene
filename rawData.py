# Raw data is downloaded and stored in rawData directory.
# From there, data is red and interpreted into numpy-arrays, which are stored in npyData directory.
# From there, data is fed into the model.


# Aktuelle Daten (Radar nur binär): https://opendata.dwd.de/weather/nwp/
# Historische Daten (aber keine Modellvorhersagen): ftp://ftp-cdc.dwd.de/pub/CDC/

import os
import urllib.request
from ftplib import FTP



thisDir = os.path.dirname(os.path.abspath(__file__))
rawDataDir = thisDir + "/rawData/"
dwdFtpServer = "ftp://ftp-cdc.dwd.de/"
radolanPath = "pub/CDC/grids_germany/hourly/radolan/recent/asc/"
dwdODServer = "https://opendata.dwd.de/"
cosmoD2Path = "weather/nwp/cosmo-d2/grib/"



def httpDownloadFile(serverName, path, fileName, targetDir):
    fullUrl = serverName + path + fileName
    fullFile = targetDir + fileName
    print("Now attempting connection to {}".format(fullUrl))
    with urllib.request.urlopen(fullUrl) as response:
        with open(fullFile, 'wb') as fileHandle:
            print("Now saving data in {}".format(fullFile))
            data = response.read()  # a bytes-object
            fileHandle.write(data)


def ftpDownloadFile(serverName, path, fileName, targetDir):
    fullFile = targetDir + fileName
    print("Now attempting connection to {}".format(serverName))
    with FTP(serverName) as ftp:
        with open(fullFile, 'wb') as fileHandle:
            ftp.login()
            ftp.cwd(path)
            print("Now saving data in {}".format(fullFile))
            tfp.retrbinary(fileName, fileHandle.write)
        


def downloadRadar(date):
    pass


def downloadModel(date):
    pass


def getModelData(fromTime, toTime, bbox, parameters):
    pass 


def getRadarData(fromTime, toTime, bbox, parameters):
    pass



if __name__ == "__main__":
    httpDownloadFile(dwdODServer, cosmoD2Path + "00/clc/", "cosmo-d2_germany_regular-lat-lon_model-level_2018111600_001_52_CLC.grib2.bz2", rawDataDir) 
    ftpDownloadFile(dwdFtpServer, radolanPath, "RW-20180101.tar.gz", rawDataDir)
