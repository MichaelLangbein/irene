import os
import numpy as np
import datetime as dt
import wradlib as wrl
import matplotlib.pyplot as plt
import plotting as p


""" 
    downloaded from ftp://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/ 
    read with https://docs.wradlib.org/en/stable/notebooks/radolan/radolan_format.html
""" 


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


def plotRadolanData(data, attrs, clabel=None):
    grid = wrl.georef.get_radolan_grid(*data.shape)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, aspect='equal')
    x = grid[:,:,0]
    y = grid[:,:,1]
    pm = ax.pcolormesh(x, y, data, cmap='viridis')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label(clabel)
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.title('{0} Product\n{1}'.format(attrs['producttype'],
                                       attrs['datetime'].isoformat()))
    plt.xlim((x[0,0],x[-1,-1]))
    plt.ylim((y[0,0],y[-1,-1]))
    plt.grid(color='r')
    plt.show()


def getDaysData(date: dt.date):
    startTime = dt.datetime(date.year, date.month, date.day)
    dataList = []
    attrList = []
    for minute in range(0, 24*60, 5):
        time = startTime + dt.timedelta(minutes=minute)
        data, attrs = readRadolanFile(time)
        data[data==-9999] = 0 #np.NaN
        dataList.append(data)
        attrList.append(attrs)
    return dataList, attrList


def splitDay(date: dt.date):
    dataL, attrL = getDaysData(date)
    films = []
    for x in range(11 * 2 - 1):
        films.append([])
        for y in range(9 * 2 - 1):
            filmData = np.zeros((288, 100, 100))
            for m5 in range(288):
                xstart = x * 50
                ystart = y * 50
                filmData[m5, :, :] = dataL[m5][xstart : xstart + 100, ystart : ystart + 100]
            films[x].append(filmData)
    return films


def analyseDay(day: dt.date):
    films = splitDay(day)
    for row in films:
        for film in row: 
            pass