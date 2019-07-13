import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random as rdm
import datetime as dt
import wradlib as wrl
import matplotlib.pyplot as plt
import h5py as h5
import pickle
import concurrent.futures
import tensorflow.keras as k
import plotting as p
from typing import List, Tuple, Union, Optional


"""
    downloaded from ftp://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/ 
    read with https://docs.wradlib.org/en/stable/notebooks/radolan/radolan_format.html
    units: https://www.dwd.de/DE/leistungen/radolan/produktuebersicht/radolan_produktuebersicht_pdf.pdf?__blob=publicationFile&v=6
    format: https://www.dwd.de/DE/leistungen/radarklimatologie/radklim_kompositformat_1_0.pdf?__blob=publicationFile&v=1
"""


rawDataDir = os.getcwd() + "/rawData"
processedDataDir = os.getcwd() + "/processedData"
frameHeight = frameWidth = frameLength = 100
frameOffset = 50
maxWorkers=2


class Frame:
    def __init__(self, data: np.array, attrs, tlIndx: Tuple = (0, 0)):
        self.data = data
        self.attrs = attrs
        self.time = attrs['datetime']
        self.tlIndx = tlIndx
        self.labels = {}

    def getSubframe(self, indices: Tuple) -> 'Frame':
        (rfOld, cfOld) = self.tlIndx
        ((rf, rt), (cf, ct)) = indices
        newData = self.data[rf:rt, cf:ct]
        newTlIndx = (rfOld + rf, cfOld + cf)
        return Frame(newData, self.attrs, newTlIndx)

    def getId(self) ->str:
        return f"frame_{self.time}_{self.tlIndx}"


class Film:
    def __init__(self, frames: List[Frame]):
        self.frames = frames
        self.frames.sort(key=lambda i: i.time)

    def append(self, frame: Frame):
        if len(self.frames) == 0:
            self.frames.append(frame)
        else:
            if frame.tlIndx == self.frames[0].tlIndx:
                startTime, endTime = self.getTimeRange()
                if frame.time == endTime + dt.timedelta(minutes=5):
                    self.frames.append(frame)
                    self.frames.sort(key=lambda i: i.time)

    def getNpData(self) -> np.array:
        T = len(self.frames)
        H, W = self.frames[0].data.shape
        data = np.zeros((T, H, W))
        for t in range(T):
            data[t, :, :] = self.frames[t].data
        return data

    def getFrameByTime(self, time: dt.datetime) -> Optional[Frame]:
        for frame in self.frames:
            if frame.time == time:
                return frame
        return None

    def getTimeRange(self) -> Tuple[dt.datetime, dt.datetime]:
        times = [frame.time for frame in self.frames]
        minTime = min(times)
        maxTime = max(times)
        return (minTime, maxTime)

    def getId(self):
        fromTime, toTime = self.getTimeRange()
        tlIndx = self.frames[0].tlIndx
        return f"storm_{int(fromTime.timestamp())}_{int(toTime.timestamp())}_{tlIndx[0]}_{tlIndx[1]}"

    @staticmethod
    def parseId(filename: str):
        parts = filename.split("_")
        if len(parts) < 4 or parts[0] != "storm":
             return None, None, None
        fromTime = dt.datetime.fromtimestamp(int(parts[1]))
        toTime = dt.datetime.fromtimestamp(int(parts[2]))
        tlIndx = (parts[3], parts[4])
        return tlIndx, fromTime, toTime


def getRadarFileName(date: dt.datetime) -> str:
    monthString = date.strftime("%Y%m")
    dayString = date.strftime("%Y%m%d")
    hourString = date.strftime("%H%M")
    ydhString = date.strftime("%y%m%d%H%M")
    fullPath = f"{rawDataDir}/YW2017.002_{monthString}/YW2017.002_{dayString}/raa01-yw2017.002_10000-{ydhString}-dwd---bin"
    return fullPath


def readRadolanFile(date: dt.datetime):
    fileName = getRadarFileName(date)
    data, attrs = wrl.io.read_radolan_composite(fileName)
    return data, attrs


def getDayData(date: dt.date):
    startTime = dt.datetime(date.year, date.month, date.day, 10)
    endTime = dt.datetime(date.year, date.month, date.day, 22)
    dataList = []
    attrList = []
    time = startTime
    while time < endTime:
        data, attrs = readRadolanFile(time)
        data[data == -9999] = 0  # np.NaN
        dataList.append(data)
        attrList.append(attrs)
        time += dt.timedelta(minutes=5)
    return dataList, attrList


def getDayFrames(date: dt.date) -> List[Frame]:
    dataL, attrL = getDayData(date)
    frames: List[Frame] = []
    for i in range(len(dataL)):
        frames.append(Frame(dataL[i], attrL[i]))
    return frames


def splitDay(date: dt.date):
    print(f"reading and splitting day {date}")
    frames = getDayFrames(date)
    H, W = frames[0].data.shape
    r = 0
    c = 0
    films: List[Film] = []
    while r + frameHeight <= H:
        while c + frameWidth <= W:
            film = Film([])
            for frame in frames:
                subframe = frame.getSubframe(((r, r + frameHeight), (c, c + frameWidth)))
                film.append(subframe)
            films.append(film)
            c += frameOffset
            print(f"just split {r}/{c}")
        r += frameOffset
        c = 0
    return films


def hatStarkregen(series: List[Frame], time: dt.datetime) -> bool:
    """
    Starkregen
    15 bis 25 l/m² in 1 Stunde
    20 bis 35 l/m² in 6 Stunden
    """

    toTime = time
    fromTime = time - dt.timedelta(hours=6)
    lastSixHours = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    sixHourSum = np.sum([el.data for el in lastSixHours], axis=0)

    toTime = time
    fromTime = time - dt.timedelta(hours=1)
    lastHour = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    lastHourSum = np.sum([el.data for el in lastHour], axis=0)

    shortTerm = (25.0 >= np.max(lastHourSum) >= 15.0)
    longTerm = (35.0 >= np.max(sixHourSum) >= 20.0)
    if (shortTerm or longTerm):
        print(f"{lastHour[-1].getId()} hat starkregen bei bei {int(lastHourSum.max())} mm/1h  und  {int(sixHourSum.max())} mm/6h")
    return (shortTerm or longTerm)


def hatHeftigerStarkregen(series: List[Frame], time: dt.datetime) -> bool:
    """
    25 bis 40 l/m² in 1 Stunde
    35 bis 60 l/m² in 6 Stunden
    """

    toTime = time
    fromTime = time - dt.timedelta(hours=6)
    lastSixHours = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    sixHourSum = np.sum([el.data for el in lastSixHours], axis=0)

    toTime = time
    fromTime = time - dt.timedelta(hours=1)
    lastHour = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    lastHourSum = np.sum([el.data for el in lastHour], axis=0)

    shortTerm = (40.0 >= np.max(lastHourSum) > 25.0)
    longTerm = (60.0 >= np.max(sixHourSum) > 35.0)
    if (shortTerm or longTerm):
        print(f"{lastHour[-1].getId()} hat heftigen starkregen bei {int(lastHourSum.max())} mm/1h  und  {int(sixHourSum.max())} mm/6h")
    return (shortTerm or longTerm)


def hatExtremerStarkregen(series: List[Frame], time: dt.datetime) -> bool:
    """
    > 40 l/m² in 1 Stunde
    > 60 l/m² in 6 Stunden
    """

    toTime = time
    fromTime = time - dt.timedelta(hours=6)
    lastSixHours = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    sixHourSum = np.sum([el.data for el in lastSixHours], axis=0)

    toTime = time
    fromTime = time - dt.timedelta(hours=1)
    lastHour = list(filter(lambda el: (fromTime <= el.time <= toTime), series))
    lastHourSum = np.sum([el.data for el in lastHour], axis=0)

    shortTerm = (np.max(lastHourSum) >= 40.0)
    longTerm = (np.max(sixHourSum) >= 60.0)
    if (shortTerm or longTerm):
        print(f"{lastHour[-1].getId()} hat extremen starkregen bei {int(lastHourSum.max())} mm/1h  und  {int(sixHourSum.max())} mm/6h")
    return (shortTerm or longTerm)


def analyseFilm(film: Film):
    for frame in film.frames:
        time = frame.time
        hatSr = hatStarkregen(film.frames, time)
        hatHeftigerSr = hatHeftigerStarkregen(film.frames, time)
        hatExtremerSr = hatExtremerStarkregen(film.frames, time)
        frame.labels = {
            "hatStarkregen": hatSr,
            "hatHeftigerSr": hatHeftigerSr,
            "hatExtremerSr": hatExtremerSr
        }


def extractStorms(film: Film, threshold) -> List[Film]:
    storms: List[Film] = []
    ongoing = False
    for frame in film.frames:
        if np.max(frame.data) > threshold:
            if not ongoing:
                ongoing = True
                storm = Film([frame])
                storms.append(storm)
            else:
                storm.append(frame)
        else:
            ongoing = False
    return storms


def analyseDay(date: dt.datetime):
    print(f"analysing day {date}")
    films = splitDay(date)
    storms: List[Film] = []
    for film in films:
        storms += extractStorms(film, 0.1)
    for storm in storms:
        analyseFilm(storm)
    stormsFltr: List[Film] = []
    for storm in storms:
        if worthSaving(storm):
            stormsFltr.append(storm)
    return stormsFltr


def worthSaving(storm) -> bool:
    for frame in storm.frames:
        if frame.labels["hatStarkregen"] or frame.labels["hatHeftigerSr"] or frame.labels["hatExtremerSr"]:
            return True
    else:
        if np.random.rand() > 0.95:
            return True
    print(f"storm {storm.getId()} not worth saving")
    return False


def saveStormToFile(storm):
    filename = f"{processedDataDir}/{storm.getId()}.pkl"
    print(f"saving storm to file {filename}")
    with open(filename, "wb") as f:
        pickle.dump(storm, f)


def loadStormFromFile(fileName: str) -> Film:
    with open(fileName, "rb") as f:
        storm = pickle.load(f)
    return storm


def analyseAndSaveDay(date):
    print(f"analysing and saving day {date}")
    storms = analyseDay(date)
    for storm in storms: 
        saveStormToFile(storm)


def analyseAndSaveTimeRange(fromTime, toTime):
    
    # get days to work with
    days = []
    time = fromTime
    while time < toTime:
        days.append(time)
        time += dt.timedelta(days=1)
    
    # execute in threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=maxWorkers) as executor:
        executor.map(analyseAndSaveDay, days)


def stormHasLabel(storm: Film, label: str):
    for frame in storm.frames:
        if frame.labels[label]:
            return True
    return False


def stormToNp(storm: Film, T: int) -> Tuple[np.array, np.array]:

    Tstorm = len(storm.frames)
    H, W = storm.frames[0].data.shape
    outData = np.zeros((T, H, W, 1))

    if T >= Tstorm:
        offset = T - Tstorm
        for t in range(Tstorm):
            outData[t + offset, :, :, 0] = storm.frames[t].data
    else:
        offset = Tstorm - T
        for t in range(T):
            outData[t, :, :, 0] = storm.frames[t + offset].data

    if stormHasLabel(storm, "hatExtremerSr"):
        outLabels = [0, 0, 0, 1]
    elif stormHasLabel(storm, "hatHeftigerSr"):
        outLabels = [0, 0, 1, 0]
    elif stormHasLabel(storm, "hatStarkregen"):
        outLabels = [0, 1, 0, 0]
    else:
        outLabels = [1, 0, 0, 0]

    return outData, outLabels


class DataGenerator(k.utils.Sequence):
    def __init__(self, dataDir, startDate: dt.datetime, endDate: dt.datetime, batchSize, timeseriesLength):
        self.dataDir = dataDir
        self.startDate = startDate
        self.endDate = endDate
        self.batchSize = batchSize
        self.timeseriesLength = timeseriesLength
        self.fileNames = self.getFileNames(dataDir, startDate, endDate)
        self.shuffleFileOrder()

    def __len__(self):
        return int(np.floor(len(self.fileNames) / self.batchSize))

    def __getitem__(self, batchNr):
        fileNames = self.fileNames[batchNr * self.batchSize : (batchNr + 1) * self.batchSize]

        dataPoints = np.zeros((self.batchSize, self.timeseriesLength, frameWidth, frameHeight, 1))
        labels = np.zeros((self.batchSize, 4))

        for bNr, fileName in enumerate(fileNames): 
            storm = loadStormFromFile(fileName)
            x, y = stormToNp(storm, self.timeseriesLength)
            dataPoints[bNr, :, :, :, :] = x
            labels[bNr, :] = y
        
        return dataPoints, labels

    def getFileNames(self, dataDir, startDate, endDate):
        filteredNames = []
        fileNames = [f for f in listdir(processedDataDir) if isfile(join(processedDataDir, f))]
        for fileName in fileNames: 
            tlIndx, fromTime, toTime = Film.parseId(fileName.strip(".pkl"))
            if tlIndx and fromTime and toTime:
                if startDate < fromTime < endDate and startDate < toTime < endDate:
                    filteredNames.append(processedDataDir + "/" + fileName)
        return filteredNames

    def shuffleFileOrder(self):
        rdm.shuffle(self.fileNames)



if __name__ == '__main__':
    fromTime = dt.date(2016, 7, 1)
    toTime = dt.date(2016, 7, 2)
    analyseAndSaveTimeRange(fromTime, toTime)
    fromTime = dt.date(2016, 6, 1)
    toTime = dt.date(2016, 6, 1)
    analyseAndSaveTimeRange(fromTime, toTime)

