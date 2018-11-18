# Tries to get data from directory of presaved npy-arrays.
# If data cannot be found there, calls rawData.getRadarData() to fetch the raw data.

import datetime as dt
import rawData as rd


def hasStorm(image):
    return 0

def hasRain(image):
    return 0


def analyseImage(image):
    labels = []
    labels.append(hasStorm(image))
    labels.append(hasRain(image))
    return labels


def getLabeledTimeseries(fromTime, toTime):
    labeledSeries = {"data": [], "labels": []}
    series = rd.getRadarData(fromTime, toTime)
    for time in series:
        image = series[time]
        labels = analyseImage(image)
        labeledSeries["data"].append(image)
        labeledSeries["labels"].append(labels)
    return labeledSeries