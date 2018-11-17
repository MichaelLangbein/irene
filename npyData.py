# Tries to get data from directory of presaved npy-arrays.
# If data cannot be found there, calls rawData.getRadarData() to fetch the raw data.

import datetime as dt
import rawData as rd


def getTimeSeries(timeFrom: dt.datetime, timeTo: dt.datetime):
    series = rd.getRadarData(timeFrom, timeTo)


def analyseImage(image):
    pass


def getLabeledTimeseries(timeFrom, timeTo):
    labeledSeries = {"data": [], "labels": []}
    series = getTimeSeries(timeFrom, timeTo)
    for t, image in enumerate(series): 
        labels = analyseImage(image)
        labeledSeries["data"].append(image)
        labeledSeries["labels"].append(labels)
    return labeledSeries