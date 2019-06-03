import os
import tensorflow.keras as k
import datetime as dt
from radarData import getOverlappingLabeledTimeseries
from plotting import movie


thisDir = os.path.dirname(os.path.abspath(__file__))
tfDataDir = thisDir + "/tfData/"
modelName = "simpleRadPredModel_checkpoint.h5"


model = k.models.load_model(tfDataDir + modelName)
dataIn, dataOut = getOverlappingLabeledTimeseries(41, dt.datetime(2016, 6, 12), dt.datetime(2016, 6, 13), 15, 1)
prediction = model.predict(dataIn)

for r, row in enumerate(prediction):
    print("----{}----".format(r))
    print("Pred: {}".format(row))
    print("Act:  {}".format(dataOut[r]))
    movie(dataIn[r], dataOut[r])