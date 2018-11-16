# Raw data is downloaded and stored in rawData directory.
# From there, data is red and interpreted into numpy-arrays, which are stored in npyData directory.
# From there, data is fed into the model.


# Aktuelle Daten (Radar nur binär): https://opendata.dwd.de/weather/nwp/
# Historische Daten (aber keine Modellvorhersagen): ftp://ftp-cdc.dwd.de/pub/CDC/

import os
import datetime as dt
import urllib.request
from ftplib import FTP



thisDir = os.path.dirname(os.path.abspath(__file__))
rawDataDir = thisDir + "/rawData/"
dwdFtpServer = "ftp-cdc.dwd.de"
radolanPath = "pub/CDC/grids_germany/hourly/radolan/recent/asc/"
dwdODServer = "https://opendata.dwd.de/"
cosmoD2Path = "weather/nwp/cosmo-d2/grib/"



def httpDownloadFile(serverName, path, fileName, targetDir):
    """ 
    >>> httpDownloadFile(dwdODServer, cosmoD2Path + "00/clc/", "cosmo-d2_germany_regular-lat-lon_model-level_2018111600_001_52_CLC.grib2.bz2", rawDataDir) 
    """
    fullUrl = serverName + path + fileName
    fullFile = targetDir + fileName
    print("Now attempting connection to {}".format(fullUrl))
    with urllib.request.urlopen(fullUrl) as response:
        with open(fullFile, "wb") as fileHandle:
            print("Now saving data in {}".format(fullFile))
            data = response.read()  # a bytes-object
            fileHandle.write(data)


def ftpDownloadFile(serverName, path, fileName, targetDir):
    """
    >>> ftpDownloadFile(dwdFtpServer, radolanPath, "RW-20180101.tar.gz", rawDataDir)
    """
    fullFile = targetDir + fileName
    print("Now attempting connection to {}".format(serverName))
    with FTP(serverName) as ftp:
        with open(fullFile, "wb") as fileHandle:
            ftp.login()
            print("Now moving to path {}".format(path))
            ftp.cwd(path)
            print("Now saving data in {}".format(fullFile))
            ftp.retrbinary("RETR " + fileName, fileHandle.write)
        

def getRadarFileName(date: dt.datetime):
    fileName = "RW-{}.tar.gz".format(date.strftime("%Y%m%d"))
    return fileName


def downloadRadar(date: dt.datetime):
    fileName = getRadarFileName(date)
    ftpDownloadFile(dwdFtpServer, radolanPath, fileName, rawDataDir)


def getModelFileName(date: dt.datetime, parameter: str, nr1: int, nr2: int):
    timeString = date.strftime("%Y%m%d%H")
    paraCap = parameter.upper()
    nr1Padded = str(nr1).zfill(3)
    nr2Padded = str(nr2).zfill(2)
    fileName = "cosmo-d2_germany_regular-lat-lon_model-level_{}_{}_{}_{}.grib2.bz2".format(timeString, nr1Padded, nr2Padded, paraCap)
    return fileName


def downloadModel(date: dt.datetime, parameter: str, nr1: int, nr2: int):
    """
    >>> downloadModel(dt.datetime(2018, 11, 16), "clc", 1, 52)
    """
    # todo: finde heraus, wofür nr1 und nr2 stehen
    # todo: assert taht date is today (openData only has todays data)
    # todo: assert that date in [0, 3, 6, ...]
    # todo: assert that parameter in [...]
    hourString = date.strftime("%H")
    fullPath = "{}/{}/{}/".format(cosmoD2Path, hourString, parameter)
    fileName = getModelFileName(date, parameter, nr1, nr2)
    httpDownloadFile(dwdODServer, fullPath, fileName, rawDataDir) 


def getModelData(fromTime, toTime, bbox, parameters):
    data = []
    timeSteps = []
    for parameter in parameters:
        for time in timeSteps:
            fileName = getModelFileName(time, parameter, 1, 1)
            #if not fileExist(fileName):
            #    downloadModel(time, parameter, 1, 1)
            #data[parameter][time] = extractModelData(file)
    return data


def getRadarData(fromTime, toTime, bbox):
    data = []
    timeSteps = []
    for time in timeSteps:
        fileName = getRadarFileName(time)
        #if not fileExists(fileName):
        #    downloadRadar(time)
        #data[time] = extractRadarData(fileName)


