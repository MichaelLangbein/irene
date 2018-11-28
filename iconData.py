dwdODServer = "https://opendata.dwd.de/"
cosmoD2Path = "weather/nwp/cosmo-d2/grib/"



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



def modelDataToNpy(date: dt.datetime, parameter: str, nr1: int, nr2: int):
    """ reads out already donwloaded and extracted grib2 data into numpy array """
    pass



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


