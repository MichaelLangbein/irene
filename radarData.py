# Raw data is downloaded and stored in rawData directory : download*Data()
# From there, data is red and interpreted into numpy-arrays, which are stored in npyData directory: extract*Data()
# From there, data is fed into the model


# Aktuelle Daten (Radar nur binaer): https://opendata.dwd.de/weather/nwp/
# Historische Daten (aber keine Modellvorhersagen): ftp://ftp-cdc.dwd.de/pub/CDC/

# TODO: wie umgehen mit fehlenden Daten?

from typing import List, Tuple
import collections as c
from utils import extract, httpDownloadFile, MyFtpServer, tprint
import config as conf
import os
import numpy as np
import time
import datetime as dt
import ftplib
import tensorflow as tf
import threading
import h5py as h5
import plotting as pl
import concurrent.futures as cf


thisDir = os.path.dirname(os.path.abspath(__file__))
rawDataDir = thisDir + "/rawData/"
npyDataDir = thisDir + "/npyData/"
dwdFtpServer = "ftp-cdc.dwd.de"
#radolanPath = "pub/CDC/grids_germany/hourly/radolan/recent/asc/"
radolanPathHistoric = "pub/CDC/grids_germany/hourly/radolan/historical/asc/"

ftpServer = MyFtpServer(conf.dwdFtpServerName, conf.dwdFtpServerUser, conf.dwdFtpServerPass, conf.ftpProxy)


class RadarFrame:
    def __init__(self, time: dt.datetime, data: np.array, lowerLeft: list, pixelSize: float):
        self.time = time
        self.data = data
        self.lowerLeft = lowerLeft
        self.pixelSize = pixelSize
        self.labels: List = []

    def getMaximumWithIndex(self) -> Tuple[int, int, int]:
        maxX, maxY = np.unravel_index(np.argmax(self.data, axis=None), self.data.shape)
        maxV = self.data[maxX, maxY]
        return (maxV, maxX, maxY)

    def getMaximumWithCoords(self) -> Tuple[int, float, float]:
        maxV, maxX, maxY = self.getMaximumWithIndex()
        maxXC, maxYC = self.getCoordsOfIndex(maxX, maxY)
        return (maxV, maxXC, maxYC)

    def getCoordsOfIndex(self, x, y) -> Tuple[float, float]:
        cX0 = self.lowerLeft[0]
        cY0 = self.lowerLeft[1]
        cX = cX0 + self.pixelSize * x
        cY = cY0 + self.pixelSize * y
        return (cX, cY)

    def getIndexOfCoords(self, cX, cY) -> Tuple[int, int]:
        cX0 = self.lowerLeft[0]
        cY0 = self.lowerLeft[1]
        x = (cX - cX0) / self.pixelSize
        y = (cY - cY0) / self.pixelSize
        return (x, y)

    def getCoordsOfCenter(self) -> Tuple[float, float]:
        X, Y = self.data.shape
        xCenter = int(X / 2)
        yCenter = int(Y / 2)
        return self.getCoordsOfIndex(xCenter, yCenter)

    def containsCoords(self, cX: float, xY: float) -> bool:
        x, y = self.getIndexOfCoords(cX, xY)
        xMax, yMax = self.data.shape
        return (x < xMax) and (y < yMax)
    
    def cropAroundIndex(self, x, y, w):
        """ also updates metadata, so that coordinate-calculation doesn't go wrong """
        xf, xt, yf, yt = self.getIndicesAroundIndex(x, y, w)

        xLL, yLL = self.getCoordsOfIndex(xf, yf)
        newData = np.copy(self.data[xf:xt, yf:yt])

        newFrame = RadarFrame(self.time, newData, [xLL, yLL], self.pixelSize)
        return newFrame
    
    def cropedAroundCoords(self, cX, cY, w):
        x, y = self.getIndexOfCoords(cX, cY)
        return self.cropAroundIndex(x, y, w)

    def getIndicesAroundIndex(self, x: int, y: int, w: int) -> Tuple[int, int, int, int]:
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
        return (xf, xt, yf, yt)

    def getIndicesAroundCoords(self, cX: float, cY: float, W: float) -> Tuple[int, int, int, int]:
        x, y = self.getIndexOfCoords(cX, cY)
        w = int(W / self.pixelSize)
        return self.getIndicesAroundIndex(x, y, w)



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
    tprint("searching for {}".format(fileName))
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
    data = []
    fromTime = fromTime.replace(minute=50)
    timeSteps = getTimeSteps(fromTime, toTime, deltaHours)
    for time in timeSteps:
        try: 
            data.append(getRadarDataForTime(time))
        except: 
            print("Something went wrong at {}".format(time))
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


def analyzeTimestep(series: List[RadarFrame], time: dt.datetime) -> List[bool]:
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


class Storm(): 

    def __init__(self, frames: List[RadarFrame]):
        self.frames = frames
        self.frames.sort(key = lambda i: i.time)

    def contains(self, cX: float, cY: float) -> bool: 
        if self.frames[0].containsCoords(cX, cY):
            return True
        return False

    def getTimeRange(self) -> Tuple[dt.datetime, dt.datetime]:
        times = [frame.time for frame in self.frames]
        minTime = min(times)
        maxTime = max(times)
        return (minTime, maxTime)

    def beforeOrAfter(self, time: dt.datetime) -> bool:
        deltaT = dt.timedelta(hours=1)
        fromTime, toTime = self.getTimeRange()
        if  (time + deltaT) == fromTime or (time - deltaT) == toTime:
            return True
        return False

    def fitsIn(self, frame: RadarFrame) -> bool:
        cX, cY = frame.getCoordsOfCenter()
        time = frame.time
        return self.contains(cX, cY) and self.beforeOrAfter(time)

    def takeOutData(self, frame: RadarFrame):
        if self.fitsIn(frame):
            cX, cY = self.frames[0].getCoordsOfCenter()
            X, Y = self.frames[0].data.shape
            croppedFrame = frame.cropedAroundCoords(cX, cY, X)
            self.frames.append(croppedFrame)
            self.frames.sort(key = lambda i: i.time)

    def applyToFrames(self, func):
        for frame in self.frames: 
            func(frame)

    def appendData(self, frame: RadarFrame): 
        if self.fitsIn(frame):
            self.frames.append(frame)
            self.frames.sort(key = lambda i: i.time)

    def prependZeroes(self, n: int):
        firstFrame = self.frames[0]
        for m in range(n):
            time = firstFrame.time - dt.timedelta(hours=(m+1))
            print("padding zeros at time {}".format(time))
            data = np.zeros(firstFrame.data.shape)
            lowerLeft = firstFrame.lowerLeft
            pixelSize = firstFrame.pixelSize
            labels = [0] * len(firstFrame.labels)
            zeroFrame = RadarFrame(time, data, lowerLeft, pixelSize)
            zeroFrame.labels = labels
            self.frames.append(zeroFrame)
        self.frames.sort(key = lambda i: i.time)



def getStorms(fromTime: dt.datetime, toTime: dt.datetime, imageSize: int, normalizer = 500, threshold = 50) -> List[Storm]:
    
    radarData = getRadarData(fromTime, toTime)
    storms: List[Storm] = []

    for frame in radarData: 
        maxval, x, y = frame.getMaximumWithIndex()
        while maxval > threshold: 

            cX, cY = frame.getCoordsOfIndex(x, y)
            croppedFrame = frame.cropedAroundCoords(cX, cY, imageSize)
            
            matchingStorms = [storm for storm in storms if storm.fitsIn(croppedFrame)]
            for storm in matchingStorms: 
                #storm.takeOutData(frame)
                storm.appendData(croppedFrame)
            if not matchingStorms:
                storms.append(Storm([croppedFrame]))
            
            xf, xt, yf, yt = frame.getIndicesAroundIndex(x, y, imageSize)
            frame.data[xf:xt, yf:yt] = np.zeros((imageSize, imageSize))
            maxval, x, y = frame.getMaximumWithIndex()
            
    return storms


def analyseStorm(storm: Storm): 
    for frame in storm.frames: 
        labels = analyzeTimestep(storm.frames, frame.time)
        frame.labels = labels


def saveStormsToFile(fileName: str, storms: List[Storm]):
    if os.path.isfile(fileName):
        raise Exception("File {} already exists".format(fileName))

    fromTime, toTime = storms[0].getTimeRange()
    for storm in storms:
        sFromTime, sToTime = storm.getTimeRange()
        if sFromTime < fromTime:
            fromTime = sFromTime
        if sToTime > toTime:
            toTime = sToTime
    
    with h5.File(fileName, 'w') as f:
        f.attrs["fromTime"] = fromTime.timestamp()
        f.attrs["toTime"] = toTime.timestamp()
        for i, storm in enumerate(storms):
            groupName = "storm_" + str(i)
            group = f.create_group(groupName)
            fromTime, toTime = storm.getTimeRange()
            group.attrs["fromTime"] = fromTime.timestamp()
            group.attrs["toTime"] = toTime.timestamp()
            group.attrs["lowerLeft"] = storm.frames[0].lowerLeft
            group.attrs["pixelSize"] = storm.frames[0].pixelSize
            for j, frame in enumerate(storm.frames):
                dsetName = "frame_" + str(j)
                dset = f.create_dataset("{}/{}".format(groupName, dsetName), data=frame.data)
                dset.attrs["labels"] = frame.labels
                dset.attrs["time"] = frame.time.timestamp()


def threadedGetAndAnalyseStorms(fromTime: dt.datetime, toTime: dt.datetime, imageSize: int): 
    tprint("now working on timerange {} to {}".format(fromTime, toTime))
    threadStorms = getStorms(fromTime, toTime, imageSize)
    for storm in threadStorms: 
            analyseStorm(storm)
    return threadStorms


def getAnalyseAndSaveStorms(fileName: str, fromTime: dt.datetime, toTime: dt.datetime, imageSize: int, nrThreads: int = 4):

    fromTimes = []
    toTimes = []
    imageSizes = []
    delta = dt.timedelta(days = 1)
    batchFromTime = fromTime
    batchToTime = fromTime + delta
    while batchFromTime < toTime:
        batchFromTime += delta
        batchToTime += delta
        fromTimes.append(batchFromTime)
        toTimes.append(batchToTime)
        imageSizes.append(imageSize)

    storms: List[Storm] = []
    with cf.ThreadPoolExecutor(max_workers=nrThreads) as executor:
        results = executor.map(threadedGetAndAnalyseStorms, fromTimes, toTimes, imageSizes)
        for result in results:
            storms += result

    saveStormsToFile(fileName, storms)


def loadStormsFromFile(fileName: str) -> List[Storm]:
    storms = []
    with h5.File(fileName, 'r') as f:
        for groupName in f.keys():
            group = f[groupName]
            fromTime = dt.datetime.fromtimestamp(group.attrs["fromTime"])
            lowerLeft = group.attrs["lowerLeft"]
            pixelSize = group.attrs["pixelSize"]
            frames = []
            for dsetName in group.keys():
                dset = group[dsetName]
                time = dt.datetime.fromtimestamp(dset.attrs["time"])
                data = np.array(dset)
                labels = dset.attrs["labels"]
                frame = RadarFrame(time, data, lowerLeft, pixelSize)
                frame.labels = labels
                frames.append(frame)
            storm = Storm(frames)
            storms.append(storm)
    return storms


def npStormsFromFile(fileName: str, nrSamples: int, timeSteps: int) -> Tuple[np.array, np.array]:
    storms = loadStormsFromFile(fileName)
    frame0 = storms[0].frames[0]
    imageWidth, imageHeight = frame0.data.shape
    inpt = np.zeros([nrSamples, timeSteps, imageWidth, imageHeight, 1])
    outpt = np.zeros([nrSamples, 3])
    nrSamples = min(nrSamples, len(storms))
    for sample in range(nrSamples):
        storm = storms[sample]
        print("storm has length {} hours".format(len(storm.frames)))
        if len(storm.frames) < timeSteps:
            storm.prependZeroes(timeSteps - len(storm.frames))
        for time in range(timeSteps):
            frame = storm.frames[time]
            inpt[sample, time, :, :, 0] = frame.data
            print("assigning data from time {} with maxval {} to slot {}".format(frame.time, np.max(frame.data), time))
        outpt[sample, :] = frame.labels
    return inpt, outpt


if __name__ == "__main__":
    fromTime = dt.datetime(2017, 6, 1, 8)
    toTime = dt.datetime(2017, 6, 3, 12)
    imageSize = 81
    try:
        os.remove("processedData/test.hdf5")
    except: 
        pass
    getAnalyseAndSaveStorms("processedData/test.hdf5", fromTime, toTime, imageSize)
    inpt, outpt = npStormsFromFile("processedData/test.hdf5", 5, 15)
    for i in range(len(outpt)):
        pl.movie(inpt[i], outpt[i])

