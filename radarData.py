# Raw data is downloaded and stored in rawData directory : download*Data()
# From there, data is red and interpreted into numpy-arrays, which are stored in npyData directory: extract*Data()
# From there, data is fed into the model


# Aktuelle Daten (Radar nur binaer): https://opendata.dwd.de/weather/nwp/
# Historische Daten (aber keine Modellvorhersagen): ftp://ftp-cdc.dwd.de/pub/CDC/

# TODO: wie umgehen mit fehlenden Daten?

from typing import List, Tuple
from utils import extract, httpDownloadFile, MyFtpServer, tprint
import config as conf
import os
import pygrib as pg
import numpy as np
import time
import datetime as dt
import ftplib
import tensorflow as tf
import threading



thisDir = os.path.dirname(os.path.abspath(__file__))
rawDataDir = thisDir + "/rawData/"
npyDataDir = thisDir + "/npyData/"
dwdFtpServer = "ftp-cdc.dwd.de"
#radolanPath = "pub/CDC/grids_germany/hourly/radolan/recent/asc/"
radolanPathHistoric = "pub/CDC/grids_germany/hourly/radolan/historical/asc/"

ftpServer = MyFtpServer(conf.dwdFtpServerName, conf.dwdFtpServerUser, conf.dwdFtpServerPass, conf.ftpProxy)


class RadarFrame:
    def __init__(self, time: dt.datetime, data: np.array, bbox: list, pixelSize):
        self.time = time
        self.data = data
        self.bbox = bbox
        self.pixelSize = pixelSize
        self.labels: List = []

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
        if w < 3 or w%2 != 1:
            raise Exception("Cropping dataframe: the window's width must be uneven, so that every corner has an equal distance from the center")
        X, Y = self.data.shape
        d = int((w-1)/2)
        xf = int(x - d)
        xt = int(x + d + 1)
        yf = int(y - d)
        yt = int(y + d + 1)
        distToLeft = xf
        if distToLeft < 0:
            xf += abs(distToLeft)
            xt += abs(distToLeft)
        distToRight = X - xt
        if distToRight < 0:
            xf -= abs(distToRight)
            xt -= abs(distToRight)
        distToTop = yf
        if distToTop < 0:
            yf += abs(distToTop)
            yt += abs(distToTop)
        distToBot = Y - yt
        if distToBot < 0:
            yf -= abs(distToBot)
            yt -= abs(distToBot)
        newBbox = self.getCoordsOfIndex(xf, yf)
        self.bbox = newBbox
        newData = self.data[xf:xt, yf:yt]
        self.data = newData
        

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



def fileToRadarFrame(date: dt.datetime) -> RadarFrame:
    """ reads out already donwloaded and extracted ascii file into RadarFrame """
    fullFileName = rawDataDir + getRadarFileName(date)
    with open(fullFileName, "r") as f:
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
                data = np.zeros([int(metaData["nrows"]), int(metaData["ncols"])], dtype=np.float32)
            else:
                row = nr - 6
                for col, el in enumerate(lineData):
                    #if el == metaData["NODATA_value"]:
                    #    data[row, col] = None
                    #else:
                        data[row, col] = float(el)
    frame = RadarFrame(date, data, [int(metaData["xllcorner"]), int(metaData["yllcorner"])], int(metaData["cellsize"]))
    return frame



def getTimeSteps(fromTime: dt.datetime, toTime: dt.datetime, deltaHours: int):
    out = []
    currentTime = fromTime
    while currentTime <= toTime:
        out.append(currentTime)
        currentTime += dt.timedelta(hours=deltaHours)
    return out


def getRadarDataForTime(time: dt.datetime) -> RadarFrame:
    fileName = getRadarFileName(time)
    if os.path.isfile(rawDataDir + fileName):
        return fileToRadarFrame(time)
    else:
        archiveFileName = getRadarFileNameDayArchive(time)
        if os.path.isfile(rawDataDir + archiveFileName):
            tprint("Now extracting archive {}".format(archiveFileName))
            extract(rawDataDir, archiveFileName)
            if not os.path.isfile(rawDataDir + fileName):
                raise KeineDatenException("The file {} does not exist anywhere!".format(fileName))
            return getRadarDataForTime(time)
        else:
            archiveArchiveFileName = getRadarFileNameMonthArchive(time)
            if os.path.isfile(rawDataDir + archiveArchiveFileName):
                tprint("Now extracting archive-archive {}".format(archiveArchiveFileName))
                extract(rawDataDir, archiveArchiveFileName)
                if not os.path.isfile(rawDataDir + archiveFileName):
                    raise KeineDatenException("The file {} does not exist anywhere!".format(archiveFileName))
                return getRadarDataForTime(time)
            else:
                tprint("Could not find {}, {} or {} locally, trying to download file".format(fileName, archiveFileName, archiveArchiveFileName))
                fileName = getRadarFileNameMonthArchive(time)
                fullRadolanPathHistory = radolanPathHistoric + time.strftime("%Y") + "/"
                try:
                    ftpServer.tryDownloadNTimes(fullRadolanPathHistory, fileName, rawDataDir, 2)
                except EOFError:
                    raise KeineDatenException("Cannot download file {}".format(fullRadolanPathHistory + fileName))
                except ftplib.error_temp as e:
                    tprint("An ftp-error has occured: {}".format(e))
                    raise KeineDatenException("Cannot download file {}".format(fullRadolanPathHistory + fileName))
                if not os.path.isfile(rawDataDir + archiveArchiveFileName):
                    raise KeineDatenException("The file {} does not exist anywhere!".format(archiveArchiveFileName))
                return getRadarDataForTime(time)


def getRadarData(fromTime: dt.datetime, toTime: dt.datetime, deltaHours=1) -> List[RadarFrame]:
    """
    >>> data = getRadarData(dt.datetime(2018, 10, 14, 0, 50), dt.datetime(2018, 10, 15, 0, 0))
    >>> for time in data:
    >>>     print(time)
    >>>     print(np.max(data[time]))
    """
    # TODO: hier bietet sich multithreading an
    data = []
    fromTime = fromTime.replace(minute=50)
    timeSteps = getTimeSteps(fromTime, toTime, deltaHours)
    for time in timeSteps:
        data.append(getRadarDataForTime(time))
    return data



def hatStarkregen(series: List[RadarFrame], time: dt.datetime) -> bool:
    """
    Starkregen	
    15 bis 25 l/m² in 1 Stunde
    20 bis 35 l/m² in 6 Stunden
    """
    toTime = time
    fromTime = time - dt.timedelta(hours=6)
    lastSixHours = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    sixHourSum = np.sum([el.data for el in lastSixHours])
    lastEntry = lastSixHours[-1].data
    shortTerm = (250 >= np.max(lastEntry) >= 150)
    longTerm = (350 >= np.max(sixHourSum) >= 200)
    return (shortTerm or longTerm)



def hatHeftigerStarkregen(series: List[RadarFrame], time: dt.datetime) -> bool:
    """
    25 bis 40 l/m² in 1 Stunde
    35 bis 60 l/m² in 6 Stunden
    """
    toTime = time
    fromTime = time - dt.timedelta(hours=6)
    lastSixHours = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    sixHourSum = np.sum([el.data for el in lastSixHours])
    lastEntry = lastSixHours[-1].data
    shortTerm = (400 >= np.max(lastEntry) > 250)
    longTerm = (600 >= np.max(sixHourSum) > 350)
    return (shortTerm or longTerm)



def hatExtremerStarkregen(series: List[RadarFrame], time: dt.datetime) -> bool:
    """
    > 40 l/m² in 1 Stunde
    > 60 l/m² in 6 Stunden
    """
    toTime = time
    fromTime = time - dt.timedelta(hours=6)
    lastSixHours = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    sixHourSum = np.sum([el.data for el in lastSixHours])
    lastEntry = lastSixHours[-1].data
    shortTerm = (np.max(lastEntry) >= 400)
    longTerm = (np.max(sixHourSum) >= 600)
    return (shortTerm or longTerm)


def analyseTimestep(series: List[RadarFrame], time: dt.datetime) -> List[bool]:
    labels = []
    labels.append(hatStarkregen(series, time))
    labels.append(hatHeftigerStarkregen(series, time))
    labels.append(hatExtremerStarkregen(series, time))
    return labels


def translateLabels(labels: List[bool]) -> List[str]:
    strLabels = []
    strLabels.append( "Starkregen" if labels[0] else "" )
    strLabels.append( "Heftiger Starkregen" if labels[1] else "" )
    strLabels.append( "Extremer Starkregen" if labels[2] else "" )
    return strLabels


class KeineDatenException(Exception):
    pass


def getLabeledTimeseries(fromTime: dt.datetime, toTime: dt.datetime, deltaHours=1) -> List[RadarFrame]:
    """ fügt zu einer bestehenden zeitreihe noch labels hinzu """
    # TODO: speichere und lade eine solche timeseries als pickle
    # TODO: diese Funktion bietet sich an zum multithreaden
    series = getRadarData(fromTime, toTime, deltaHours)
    if len(series) == 0:
        raise KeineDatenException("Keine Radardaten zwischen {} und {}".format(fromTime, toTime))
    for frame in series:
        frame.labels = analyseTimestep(series, frame.time)
    return series



def cropAroundMaximum(series, size):
    """ 
    Findet position des Maximums einer Serie von RadarFrames.
    Schneidet aus jedem frame ein Fenster um dieses Maximum herum aus.
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
    for frame in series:
        frame.cropAroundCoords(maxX, maxY, size)



def normalizeToMaximum(series):
    """
    findet maximum einer serie von RadarFrames.
    normalisiert ganze serie zu diesem maximum.
    """
    pass


# def getOverlappingLabeledTimeseries(imageSize: int, fromTime, toTime, timeSteps = 10, deltaHours = 1, normalisationValue = 400) -> Tuple[np.array, np.array]:
#     """ 
#      - lädt eine labeled timeseries
#      - formattiert sie so, dass für keras brauchbar
#      - crop'en der Daten, so dass nur interessante events zu sehen
#      - TODO: Normalisieren der Daten 
#     """


#     labeledSeries = getLabeledTimeseries(fromTime, toTime, deltaHours)
#     imageWidth = imageHeight = imageSize
#     batchSize = len(labeledSeries) - timeSteps
#     dataIn = np.zeros([batchSize, timeSteps, imageWidth, imageHeight, 1])
#     dataOut = np.zeros([batchSize, 3])

#     for frame in labeledSeries: 
#         frame.data = frame.data / normalisationValue

#     for batchNr in range(batchSize):
#         subSeries = labeledSeries[batchNr:batchNr+timeSteps]
#         cropAroundMaximum(subSeries, imageSize)
#         for timeNr, frame in enumerate(subSeries):
#             dataIn[batchNr, timeNr, :, :, 0] = frame.data
#         dataOut[batchNr, :] = labeledSeries[batchNr + timeSteps].labels

#     return (dataIn, dataOut)
        

# def getRandomOverlappingBatch(batchSize, timeSteps, imageSize):
#     year = np.random.randint(2006, 2017)   
#     month = np.random.randint(4, 11)
#     day = np.random.randint(1, 27)
#     fromTime = dt.datetime(year, month, day)
#     toTime = fromTime + dt.timedelta(hours=(timeSteps + batchSize))
#    tprint("Getting data from {} to {}".format(fromTime, toTime))
#     try:
#         data, labels = getOverlappingLabeledTimeseries(imageSize, fromTime, toTime, timeSteps)
#     except (IOError, KeineDatenException):
#        tprint("Keine Daten erhalten für {} bis {}. Probiere es mit anderem Zeitraum".format(fromTime, toTime))
#         data, labels = getRandomOverlappingBatch(batchSize, timeSteps, imageSize)
#     return data, labels



def getRandomSeries(timeSteps: int) -> List[RadarFrame]:
    year = np.random.randint(2016, 2017)
    month = np.random.randint(4, 11)
    day = np.random.randint(1, 27)
    earliest = max(8 - timeSteps, 0)
    latest = 24
    hour = np.random.randint(earliest, latest)
    fromTime = dt.datetime(year, month, day, hour)
    toTime = fromTime + dt.timedelta(hours=(timeSteps))
    try:
        labeledSeries = getLabeledTimeseries(fromTime, toTime, 1)
    except (IOError, KeineDatenException):
        tprint("Keine Daten erhalten für {} bis {}. Probiere es mit anderem Zeitraum".format(fromTime, toTime))
        labeledSeries = getRandomSeries(timeSteps)
    return labeledSeries


def getRandomBatch(batchSize: int, timeSteps: int, imageSize: int, normalizer = 400) -> Tuple[np.array, np.array]:
    imageWidth = imageHeight = imageSize
    dataIn = np.zeros([batchSize, timeSteps, imageWidth, imageHeight, 1])
    dataOut = np.zeros([batchSize, 3])
    for batchNr in range(batchSize):
        labeledSeries = getRandomSeries(timeSteps)
        cropAroundMaximum(labeledSeries, imageSize)
        for timeNr, frame in enumerate(labeledSeries):
            dataIn[batchNr, timeNr, :, :, 0] = frame.data / normalizer
        dataOut[batchNr, :] = labeledSeries[-1].labels
    return (dataIn, dataOut)


def radarGenerator(batchSize, timeSteps, imageSize):
    if imageSize < 3 or imageSize%2 != 1:
        raise Exception("Image size must be uneven, so that maximum is centered!")
    while True:
        data, labels = getRandomBatch(batchSize, timeSteps, imageSize)
        yield (data, labels)
            


def threadedGetRandomSeries(timeSteps, imageSize, dIn, dOut, normalizer):
    labeledSeries = getRandomSeries(timeSteps)
    cropAroundMaximum(labeledSeries, imageSize)
    for timeNr, frame in enumerate(labeledSeries):
        dIn[timeNr, :, :, 0] = frame.data / normalizer
    dOut[:] = labeledSeries[-1].labels


def threadedGetRandomBatch(batchSize: int, timeSteps: int, imageSize: int, normalizer = 400) -> Tuple[np.array, np.array]:
    # allocating memory
    imageWidth = imageHeight = imageSize
    dataIn = np.zeros([batchSize, timeSteps, imageWidth, imageHeight, 1])
    dataOut = np.zeros([batchSize, 3])

    # setting up threads
    threads = []
    for batchNr in range(batchSize):
        thread = threading.Thread(target=threadedGetRandomSeries, args=(timeSteps, imageSize, dataIn[batchNr], dataOut[batchNr], normalizer))
        threads.append(thread)
        thread.start()
    
    # joining threads
    for thread in threads:
        thread.join()

    # return 
    return (dataIn, dataOut)

def threadedRadarGenerator(batchSize, timeSteps, imageSize):
    if imageSize < 3 or imageSize%2 != 1:
        raise Exception("Image size must be uneven, so that maximum is centered!")
    while True:
        data, labels = threadedGetRandomBatch(batchSize, timeSteps, imageSize)
        yield (data, labels)


if __name__ == "__main__":
    gen = threadedRadarGenerator(4, 10, 81)
    i = 0
    for data, label in gen:
        i += 1
        print("iteration {}".format(i))
        print(np.max(data))
        if i > 10:
            break