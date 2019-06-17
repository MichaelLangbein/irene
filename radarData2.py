import os
import numpy as np
import random as rdm
import datetime as dt
import wradlib as wrl
import matplotlib.pyplot as plt
import h5py as h5
import plotting as p
from typing import List, Tuple, Union, Optional


""" 
    downloaded from ftp://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/ 
    read with https://docs.wradlib.org/en/stable/notebooks/radolan/radolan_format.html
    units: https://www.dwd.de/DE/leistungen/radolan/produktuebersicht/radolan_produktuebersicht_pdf.pdf?__blob=publicationFile&v=6
    format: https://www.dwd.de/DE/leistungen/radarklimatologie/radklim_kompositformat_1_0.pdf?__blob=publicationFile&v=1
""" 



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
        self.frames.sort(key = lambda i: i.time)

    def append(self, frame: Frame):
        if len(self.frames) == 0:
            self.frames.append(frame)
        else:
            if frame.tlIndx == self.frames[0].tlIndx:
                startTime, endTime = self.getTimeRange()
                if frame.time == endTime + dt.timedelta(minutes=5):
                    self.frames.append(frame)
                    self.frames.sort(key = lambda i: i.time)

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
        return f"storm_{tlIndx}_{fromTime}_{toTime}"



def getRadarFileName(date: dt.datetime) -> str: 
    thisDir = os.path.abspath('')
    monthString = date.strftime("%Y%m")
    dayString = date.strftime("%Y%m%d")
    hourString = date.strftime("%H%M")
    ydhString = date.strftime("%y%m%d%H%M")
    fullPath = "{dir}/rawData2/YW2017.002_{month}/YW2017.002_{day}/raa01-yw2017.002_10000-{ydh}-dwd---bin".format(
        **{"month": monthString, "day": dayString, "hour": hourString, "ydh": ydhString,  "dir": thisDir})
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
        data[data==-9999] = 0 #np.NaN
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
    frameHeight = frameWidth = frameLength = 100
    frameOffset = 50
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
        print(f"{lastHour[-1].getId()} hat starkregen bei {lastHourSum.max()} mm/1h  und  {sixHourSum.max()} mm/6h")
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
        print(f"{lastHour[-1].getId()} hat heftigen starkregen bei {lastHourSum.max()} mm/1h  und  {sixHourSum.max()} mm/6h")
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
        print(f"{lastHour[-1].getId()} hat extremen starkregen bei {lastHourSum.max()} mm/1h  und  {sixHourSum.max()} mm/6h")
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


def filterStorms(film: Film, threshold) -> List[Film]:
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
    print(f"splitting day {date}")
    films = splitDay(date)
    storms: List[Film] = []
    for film in films: 
        storms += filterStorms(film, 0.1)
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


def appendStormsToFile(fileName, storms):
    with h5.File(fileName, 'a') as fileHandle:
        rdm.shuffle(storms) # <--- if we dont do this, small storms will be in the beginning and large ones at the end, because we only save one or two days at a time. If we then only load half the dataset, we train the net only with small storms. 
        for storm in storms:
            groupName = storm.getId()
            print(f"saving storm {groupName}")
            group = fileHandle.create_group(groupName)
            fromTime, toTime = storm.getTimeRange()
            group.attrs["fromTime"] = fromTime.timestamp()
            group.attrs["toTime"] = toTime.timestamp()
            group.attrs["tlIndx"] = storm.frames[0].tlIndx
            for key in storm.frames[0].attrs:
                group.attrs[f"attrs_{key}"] = str(storm.frames[0].attrs[key])
            for frame in storm.frames:
                dsetName = frame.getId()
                dset = fileHandle.create_dataset(f"{groupName}/{dsetName}", data=frame.data)
                dset.attrs["time"] = frame.time.timestamp()
                dset.attrs["hatStarkregen"] = frame.labels["hatStarkregen"]     
                dset.attrs["hatHeftigerSr"] = frame.labels["hatHeftigerSr"]
                dset.attrs["hatExtremerSr"] = frame.labels["hatExtremerSr"]


def analyseAndSaveTimeRange(fromTime, toTime, fileName):
    time = fromTime
    while time < toTime:
        storms = analyseDay(time)
        appendStormsToFile(fileName, storms)
        time += dt.timedelta(days=1)


def loadStormsFromFile(fileName: str, nrSamples: int, minLength: int = 1) -> List[Film]:
    storms = []
    i = 0
    with h5.File(fileName, 'r') as f:
        print(f"getting {nrSamples} storms out of {len(f.keys())} available")
        for groupName in f.keys():
            print(f"loading {groupName}")
            group = f[groupName]
            fromTime = dt.datetime.fromtimestamp(group.attrs["fromTime"])
            tlIndx = tuple(group.attrs["tlIndx"])
            frames = []
            for dsetName in group.keys():
                dset = group[dsetName]
                time = dt.datetime.fromtimestamp(dset.attrs["time"])
                data = np.array(dset)
                labels = {
                    "hatStarkregen": dset.attrs["hatStarkregen"],
                    "hatHeftigerSr": dset.attrs["hatHeftigerSr"],
                    "hatExtremerSr": dset.attrs["hatExtremerSr"]
                }
                frame = Frame(data, {'datetime': time}, tlIndx)
                frame.labels = labels
                frames.append(frame)
            if len(frames) >= minLength: 
                storm = Film(frames)
                storms.append(storm)
                i += 1
            if i >= nrSamples:
                break
    return storms


def hat(storm: Film, label: str):
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
    
    outLabels = [0, 0, 0]
    if hat(storm, "hatExtremerSr"):
        outLabels = [0, 0, 1]
    elif hat(storm, "hatHeftigerSr"):
        outLabels = [0, 1, 0]
    elif hat(storm, "hatStarkregen"):
        outLabels = [1, 0, 0]
    #outLabels = [int(val) for val in storm.frames[-1].labels.values()]

    return outData, outLabels


def loadTfData(fileName, T, maxSamples):
    storms = loadStormsFromFile(fileName, maxSamples, 4)
    dataList = []
    labelList = []
    for storm in storms: 
        data, label = stormToNp(storm, T)
        dataList.append(data)
        labelList.append(label)
    return np.array(dataList), np.array(labelList)



def redoAnalysis(fileName: str): 
    with h5.File(fileName, 'a') as f:
        for groupName in f.keys():
            group = f[groupName]
            fromTime = dt.datetime.fromtimestamp(group.attrs["fromTime"])
            tlIndx = tuple(group.attrs["tlIndx"])
            frames = []
            for dsetName in group.keys():
                dset = group[dsetName]
                time = dt.datetime.fromtimestamp(dset.attrs["time"])
                data = np.array(dset)
                frame = Frame(data, {'datetime': time}, tlIndx)
                frames.append(frame)
            film = Film(frames)
            analyseFilm(film)
            for frame in film.frames:
                frameId = frame.getId()
                dset = group[frameId]
                dset.attrs["hatStarkregen"] = frame.labels["hatStarkregen"]
                dset.attrs["hatHeftigerSr"] = frame.labels["hatHeftigerSr"]
                dset.attrs["hatExtremerSr"] = frame.labels["hatExtremerSr"]




if __name__ == '__main__':
    # fileName = "test.h5"
    # if os.path.isfile(fileName):
    #     os.remove(fileName)
    fromTime = dt.date(2016, 6, 1)
    toTime = dt.date(2016, 6, 5)
    analyseAndSaveTimeRange(fromTime, toTime, "training_2016.h5")
    #redoAnalysis("validation_2016.h5")
    dataL, labelL = loadTfData("training_2016.h5", int(5 * 60/5), 100)
    print(f"labels: {np.sum(labelL, axis=0)}")
    for indx in range(2):
        p.movie(dataL[indx, :, :, :, 0], labelL[indx], 15)
