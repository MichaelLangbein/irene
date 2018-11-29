# Raw data is downloaded and stored in rawData directory : download*Data()
# From there, data is red and interpreted into numpy-arrays, which are stored in npyData directory: extract*Data()
# From there, data is fed into the model


# Aktuelle Daten (Radar nur binär): https://opendata.dwd.de/weather/nwp/
# Historische Daten (aber keine Modellvorhersagen): ftp://ftp-cdc.dwd.de/pub/CDC/

from utils import extract, httpDownloadFile, MyFtpServer
import os
import pygrib as pg
import numpy as np
import time
import datetime as dt
import ftplib



thisDir = os.path.dirname(os.path.abspath(__file__))
rawDataDir = thisDir + "/rawData/"
npyDataDir = thisDir + "/npyData/"
dwdFtpServer = "ftp-cdc.dwd.de"
radolanPath = "pub/CDC/grids_germany/hourly/radolan/recent/asc/"
radolanPathHistoric = "pub/CDC/grids_germany/hourly/radolan/historical/asc/"

ftpServer = MyFtpServer(dwdFtpServer)


class RadarFrame:
    def __init__(self, time: dt.datetime, data: np.array, bbox: list, pixelSize):
        self.time = time
        self.data = data
        self.bbox = bbox
        self.pixelSize = pixelSize

    def getMaximumWithIndex(self):
        maxX, maxY = np.unravel_index(np.argmax(self.data, axis=None), self.data.shape)
        maxV = self.data[maxX, maxY]
        return (maxV, maxX, maxY)

    def getMaximumWithCoords(self):
        maxV, maxX, maxY = self.getMaximumWithIndex()
        maxXC, maxYC = self.getCoordsOfIndex(maxX, maxY)
        return (maxV, maxXC, maxYC)

    def getCoordsOfIndex(self, x, y):
        cX0 = self.bbox[0]
        cY0 = self.bbox[1]
        cX = cX0 + self.pixelSize * x
        cY = cY0 + self.pixelSize * y
        return (cX, cY)

    def getIndexOfCoords(self, cX, cY):
        cX0 = self.bbox[0]
        cY0 = self.bbox[1]
        x = (cX - cX0) / self.pixelSize
        y = (cY - cY0) / self.pixelSize
        return (x, y)
    
    def cropAroundIndex(self, x, y, w):
        """ also updates metadata, so that coordinate-calculation doesn't go wrong """
        pass

    def cropAroundCoords(self, cX, cY, w):
        x, y = self.getIndexOfCoords(cX, cY)
        self.cropAroundIndex(x, y, w)



def getRadarFileName(date: dt.datetime):
    fileName = "RW_{}-{}.asc".format(date.strftime("%Y%m%d"), date.strftime("%H%M"))
    return fileName



def getRadarFileNameDayArchive(date: dt.datetime):
    fileName = "RW-{}.tar.gz".format(date.strftime("%Y%m%d"))
    return fileName



def getRadarFileNameMonthArchive(date: dt.datetime):
    fileName = "RW-{}.tar".format(date.strftime("%Y%m"))
    return fileName



def radarDataToNpy(date: dt.datetime):
    """ reads out already donwloaded and extracted ascii file into numpy array """
    fullFileName = rawDataDir + getRadarFileName(date)
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
    frame = RadarFrame(date, data, [metaData["xllcorner"], metaData["yllcorner"]], metaData["cellsize"])
    return frame



def getTimeSteps(fromTime: dt.datetime, toTime: dt.datetime, deltaHours: int):
    out = []
    currentTime = fromTime
    while currentTime <= toTime:
        out.append(currentTime)
        currentTime += dt.timedelta(hours=deltaHours)
    return out



def downloadUnzipRadar(date: dt.datetime):
    """
    >>> downloadUnzipRadar(dt.datetime(2018,10,14))
    """
    try:
        fileName = getRadarFileNameDayArchive(date)
        print("Searching for file {} in recent-data-dir {}:".format(fileName, radolanPath))
        ftpServer.downloadFile(radolanPath, fileName, rawDataDir)
        extract(rawDataDir, fileName)  
    except ftplib.error_perm:
        fileName = getRadarFileNameMonthArchive(date)
        print("Searching for file {} in historical-data-dir {}:".format(fileName, radolanPathHistoric))
        fullRadolanPathHistory = radolanPathHistoric + date.strftime("%Y") + "/"
        ftpServer.downloadFile(fullRadolanPathHistory, fileName, rawDataDir)
        extract(rawDataDir, fileName)
        fileNameMonth = getRadarFileNameDayArchive(date)
        print("Now extracting sub-archive {}".format(fileNameMonth))
        extract(rawDataDir, fileNameMonth)



def getRadarData(fromTime, toTime, bbox = None):
    """
    >>> data = getRadarData(dt.datetime(2018, 10, 14, 0, 50), dt.datetime(2018, 10, 15, 0, 0))
    >>> for time in data:
    >>>     print(time)
    >>>     print(np.max(data[time]))
    """
    data = []
    fromTime = fromTime.replace(minute=50)
    timeSteps = getTimeSteps(fromTime, toTime, 3)
    for time in timeSteps:
        print("Getting data for time {}".format(time))
        fileName = getRadarFileName(time)
        fullFileName = rawDataDir + fileName
        if os.path.isfile(fullFileName):
            print("Found file {} locally".format(fullFileName))
        else:
            archiveFileName = getRadarFileNameDayArchive(time)
            if os.path.isfile(rawDataDir + archiveFileName):
                print("Found file {} locally. Extracting now.".format(archiveFileName))
                extract(rawDataDir, archiveFileName)
            else:
                archiveArchiveFileName = getRadarFileNameMonthArchive(time)
                if os.path.isfile(rawDataDir + archiveArchiveFileName):
                    print("Found file {} locally. Extracting now".format(archiveArchiveFileName))
                    extract(rawDataDir, archiveArchiveFileName)
                    extract(rawDataDir, archiveFileName)
                else:
                    print("Could not find {}, {} or {} locally, trying to download file".format(fileName, archiveFileName, archiveArchiveFileName))
                    downloadUnzipRadar(time)
        data.append(radarDataToNpy(time))
    return data



def hatStarkregen(series, time: dt.datetime):
    """
    Starkregen	
    15 bis 25 l/m² in 1 Stunde
    20 bis 35 l/m² in 6 Stunden
    """
    toTime = time
    fromTime = time.replace(hour=time.hour - 6)
    lastSixHours = filter(lambda el: (fromTime <= el.time <= toTime), series)
    sixHourSum = map(lambda carry, el: carry + el, lastSixHours)
    lastEntry = lastSixHours[-1]
    shortTerm = (250 >= np.max(lastEntry) >= 150)
    longTerm = (350 >= np.max(sixHourSum) >= 200)
    return (shortTerm or longTerm)



def hatHeftigerStarkregen(series, time: dt.datetime):
    """
    25 bis 40 l/m² in 1 Stunde
    35 bis 60 l/m² in 6 Stunden
    """
    toTime = time
    fromTime = time.replace(hour=time.hour - 6)
    lastSixHours = filter(lambda el: (fromTime <= el.time <= toTime), series)
    sixHourSum = map(lambda carry, el: carry + el, lastSixHours)
    lastEntry = lastSixHours[-1]
    shortTerm = (400 >= np.max(lastEntry) > 250)
    longTerm = (600 >= np.max(sixHourSum) > 350)
    return (shortTerm or longTerm)



def hatExtremerStarkregen(series, time: dt.datetime):
    """
    > 40 l/m² in 1 Stunde
    > 60 l/m² in 6 Stunden
    """
    toTime = time
    fromTime = time.replace(hour=time.hour - 6)
    lastSixHours = filter(lambda el: (fromTime <= el.time <= toTime), series)
    sixHourSum = map(lambda carry, el: carry + el, lastSixHours)
    lastEntry = lastSixHours[-1]
    shortTerm = (np.max(lastEntry) >= 400)
    longTerm = (np.max(sixHourSum) >= 600)
    return (shortTerm or longTerm)


def analyseTimestep(series, time: dt.datetime):
    labels = []
    labels.append(hatStarkregen(series, time))
    labels.append(hatHeftigerStarkregen(series, time))
    labels.append(hatExtremerStarkregen(series, time))
    return labels



def getLabeledTimeseries(fromTime, toTime):
    """ fügt zu einer bestehenden zeitreihe noch labels hinzu """
    # TODO: speichere und lade eine solche timeseries als pickle
    series = getRadarData(fromTime, toTime)
    rawData = map(lambda el: el.data, series)
    for frame in series:
        frame.labels = analyseTimestep(frame.time, rawData)
    return series



def cropAroundMaximum(series, size):
    """ 
    findet position des maximums einer serie von RadarFrames.
    schneidet aus jedem frame ein fenster um dieses maximum herum aus.
    """
    maximum = 0.0
    maxX = 0
    maxY = 0
    maxT = 0
    for timeStep, frame in enumerate(series):
        (localMax, localMaxX, localMaxY) = frame.getMaximumWithCoords()
        if localMax > maximum:
            maximum = localMax
            maxX = localMaxX
            maxY = localMaxY
            maxT = timeStep
    print("Now cropping series around t={}, x={}, y={}, v={}")
    for frame in series:
        frame.cropAroundCoords(maxX, maxY, size)



def normalizeToMaximum(series):
    """
    findet maximum einer serie von RadarFrames.
    normalisiert ganze serie zu diesem maximum.
    """
    pass


def getOverlappingLabeledTimeseries(imageSize, fromTime, toTime, timeSteps = 10):
    """ 
     - lädt eine labeled timeseries
     - formattiert sie so, dass für keras brauchbar
     - TODO: crop'en der Daten, so dass nur interessante events zu sehen
     - TODO: Normalisieren der Daten 
    """
    labeledSeries = getLabeledTimeseries(fromTime, toTime)
    imageWidth = imageHeight = imageSize
    batchSize = len(labeledSeries) - timeSteps
    dataIn = np.zeros([batchSize, timeSteps, imageWidth, imageHeight, 1])
    dataOut = np.zeros([batchSize, 3])
    for batchNr in range(batchSize):
        subSeries = labeledSeries[batchNr:batchNr+timeSteps]
        cropAroundMaximum(subSeries, imageWidth)
        for timeNr, frame in enumerate(subSeries):
            dataIn[batchNr, timeNr, :, :, 1] = frame.data
        dataOut[batchNr, :] = labeledSeries[batchNr + timeSteps + 1].labels
    return (dataIn, dataOut)
        
