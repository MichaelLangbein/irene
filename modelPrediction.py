import os
import tensorflow.keras as k
import datetime as dt
import radarData2 as rd
from plotting import movie


thisDir = os.path.dirname(os.path.abspath(__file__))
tfDataDir = thisDir + "/tfData/"
modelName = "latestRadPredModel.h5"


model = k.models.load_model(tfDataDir + modelName)
dataIn, dataOut = rd.loadTfData("test_0.h5", int(5 * 60/5), 5)
prediction = model.predict(dataIn)

for r, row in enumerate(prediction):
    print("----{}----".format(r))
    print("Pred: {}".format(row))
    print("Act:  {}".format(dataOut[r]))
    movie(dataIn[r, :, :, :, 0], dataOut[r], 15)