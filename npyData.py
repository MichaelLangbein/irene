# Todos:
#  - image labelling: also consinder history (6h)
#  - storms: also consider wind and hail (is part of definition from dwd:
#       "Gewitter mit Hagelschlag, heftigem Starkregen oder Orkan(artigen)Böen"
#    )

import datetime as dt
import rawData as rd
import numpy as np


def hatStarkregen(image):
    """
    Starkregen	
    15 bis 25 l/m² in 1 Stunde
    20 bis 35 l/m² in 6 Stunden
    """
    return np.max(image) > 150

def hatHeftigerStarkregen(image):
    """
    > 25 l/m² in 1 Stunde
    > 35 l/m² in 6 Stunden
    """
    return np.max(image) > 250


def hatExtemerStarkregen(image):
    """
    > 40 l/m² in 1 Stunde
    > 60 l/m² in 6 Stunden
    """
    return np.max(image) > 400


def analyseImage(image):
    labels = []
    labels.append(hatStarkregen(image))
    labels.append(hatHeftigerStarkregen(image))
    labels.append(hatExtemerStarkregen(image))
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


data = getLabeledTimeseries(dt.datetime(2016, 10, 14), dt.datetime(2016, 10, 24))
print(data["labels"])