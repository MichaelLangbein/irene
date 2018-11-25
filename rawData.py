# Raw data is downloaded and stored in rawData directory : download*Data()
# From there, data is red and interpreted into numpy-arrays, which are stored in npyData directory: extract*Data()
# From there, data is fed into the model: get*Data()


# Aktuelle Daten (Radar nur binär): https://opendata.dwd.de/weather/nwp/
# Historische Daten (aber keine Modellvorhersagen): ftp://ftp-cdc.dwd.de/pub/CDC/

import os
import bz2
import tarfile
import datetime as dt
import urllib.request
from ftplib import FTP
import ftplib
import pygrib as pg
import numpy as np
import time



thisDir = os.path.dirname(os.path.abspath(__file__))
rawDataDir = thisDir + "/rawData/"
dwdFtpServer = "ftp-cdc.dwd.de"
radolanPath = "pub/CDC/grids_germany/hourly/radolan/recent/asc/"
radolanPathHistory = "pub/CDC/grids_germany/hourly/radolan/historical/asc/"
dwdODServer = "https://opendata.dwd.de/"
cosmoD2Path = "weather/nwp/cosmo-d2/grib/"



def extract(path, fileName):
    fullName = path + fileName
    if (fullName.endswith("tar.gz")):
        with tarfile.open(fullName, "r:gz") as tar:
            tar.extractall(path)
    elif (fullName.endswith("tar")):
        with tarfile.open(fullName, "r:") as tar:
            tar.extractall(path)
    elif(fullName.endswith(".bz2")):
        extractedName = fullName[:-4]
        with bz2.BZ2File(fullName) as zipFile: # open the file
            with open(extractedName, "wb") as file:
                data = zipFile.read() # get the decompressed data
                file.write(data)



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



def ftpDownloadFile(serverName, path, fileName, targetDir, attemptNr=1):
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



def getHistoricRadarFileName(date: dt.datetime):
    fileName = "RW-{}.tar".format(date.strftime("%Y%m"))
    return fileName



def getRadarFileNameUnzipped(date: dt.datetime):
    fileName = "RW_{}-{}.asc".format(date.strftime("%Y%m%d"), date.strftime("%H%M"))
    return fileName


def getModelFileName(date: dt.datetime, parameter: str, nr1: int, nr2: int):
    timeString = date.strftime("%Y%m%d%H")
    paraCap = parameter.upper()
    nr1Padded = str(nr1).zfill(3)
    nr2Padded = str(nr2).zfill(2)
    fileName = "cosmo-d2_germany_regular-lat-lon_model-level_{}_{}_{}_{}.grib2.bz2".format(timeString, nr1Padded, nr2Padded, paraCap)
    return fileName



def getModelFileNameUnzipped(date, parameter, nr1, nr2):
    fileNameZipped = getModelFileName(date, parameter, nr1, nr2)
    return fileNameZipped[:-4]



def downloadUnzipRadar(date: dt.datetime):
    """
    >>> downloadUnzipRadar(dt.datetime(2018,10,14))
    """
    try:
        fileName = getRadarFileName(date)
        print("Searching for file {} in recent-data-dir {}:".format(fileName, radolanPath))
        ftpDownloadFile(dwdFtpServer, radolanPath, fileName, rawDataDir)
        extract(rawDataDir, fileName)  
    except ftplib.error_perm:
        fileName = getHistoricRadarFileName(date)
        print("Searching for file {} in historical-data-dir {}:".format(fileName, radolanPathHistory))
        fullRadolanPathHistory = radolanPathHistory + date.strftime("%Y") + "/"
        ftpDownloadFile(dwdFtpServer, fullRadolanPathHistory, fileName, rawDataDir)
        extract(rawDataDir, fileName)
        fileNameMonth = getRadarFileName(date)
        print("Now extracting sub-archive {}".format(fileNameMonth))
        extract(rawDataDir, fileNameMonth)



def downloadUnzipModel(date: dt.datetime, parameter: str, nr1: int, nr2: int):
    """
    >>> downloadUnzipModel(dt.datetime(2018, 11, 16), "clc", 1, 52)
    """
    # todo: finde heraus, wofür nr1 und nr2 stehen
    # todo: assert that date is today (openData only has todays data)
    # todo: assert that date in [0, 3, 6, ...]
    # todo: assert that parameter in [...]
    hourString = date.strftime("%H")
    fullPath = "{}/{}/{}/".format(cosmoD2Path, hourString, parameter)
    fileName = getModelFileName(date, parameter, nr1, nr2)
    httpDownloadFile(dwdODServer, fullPath, fileName, rawDataDir) 
    extract(rawDataDir, fileName)



def radarDataToNpy(date: dt.datetime):
    """ reads out already donwloaded and extracted ascii file into numpy array """
    fullFileName = rawDataDir + getRadarFileNameUnzipped(date)
    with open(fullFileName, "r") as f:
        print("Reading data from {}".format(fullFileName))
        metaData = {}  
        for nr, line in enumerate(f):
            lineData = line.split()
            if nr == 0: 
                metaData["ncols"] = lineData[1]
            elif nr == 1:
                metaData["nrows"] = lineData[1]
            elif nr == 2:
                metaData["xllcorner"] = lineData[1]
            elif nr == 3:
                metaData["yllcorner"] = lineData[1]
            elif nr == 4:
                metaData["cellsize"] = lineData[1]
            elif nr == 5:
                metaData["NODATA_value"] = lineData[1]
                print("Read this metadata from file:")
                print(metaData)
                data = np.zeros([int(metaData["nrows"]), int(metaData["ncols"])])
            else:
                row = nr - 6
                for col, el in enumerate(lineData):
                    #if el == metaData["NODATA_value"]:
                    #    data[row, col] = None
                    #else:
                        data[row, col] = el
    return data



def modelDataToNpy(date: dt.datetime, parameter: str, nr1: int, nr2: int):
    """ reads out already donwloaded and extracted grib2 data into numpy array """
    pass




def getTimeSteps(fromTime: dt.datetime, toTime: dt.datetime, deltaHours: int):
    out = []
    currentTime = fromTime
    while currentTime <= toTime:
        out.append(currentTime)
        currentTime += dt.timedelta(hours=deltaHours)
    return out


def getModelData(fromTime, toTime, bbox, parameters):
    data = {}
    timeSteps = getTimeSteps(fromTime, toTime, 3)
    for parameter in parameters:
        for time in timeSteps:
            fileName = getModelFileNameUnzipped(time, parameter, 1, 1)
            fullFileName = rawDataDir + fileName
            if not os.path.isfile(fullFileName):
                downloadUnzipModel(time, parameter, 1, 1)
            #data[parameter][time] = extractModelData(file)
    return data



def getRadarData(fromTime, toTime, bbox = None):
    """
    >>> data = getRadarData(dt.datetime(2018, 10, 14, 0, 50), dt.datetime(2018, 10, 15, 0, 0))
    >>> for time in data:
    >>>     print(time)
    >>>     print(np.max(data[time]))
    """
    data = {}
    fromTime = fromTime.replace(minute=50)
    timeSteps = getTimeSteps(fromTime, toTime, 3)
    for time in timeSteps:
        print("Getting data for time {}".format(time))
        fileName = getRadarFileNameUnzipped(time)
        fullFileName = rawDataDir + fileName
        if os.path.isfile(fullFileName):
            print("Found file {} locally".format(fullFileName))
        else:
            archiveFileName = getRadarFileName(time)
            if os.path.isfile(archiveFileName):
                print("Found file {} locally. Extracting now.".format(archiveFileName))
                extract(rawDataDir, archiveFileName)
            else:
                print("Could not find {} locally, trying to download file".format(fileName))
                downloadUnzipRadar(time)
        data[time] = radarDataToNpy(time)
    return data



