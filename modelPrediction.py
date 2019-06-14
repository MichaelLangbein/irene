import os
import tensorflow.keras as k
import datetime as dt
import radarData as rd
from plotting import movie


thisDir = os.path.dirname(os.path.abspath(__file__))
tfDataDir = thisDir + "/tfData/"
modelName = "latestRadPredModel.h5"


model = k.models.load_model(tfDataDir + modelName)
dataIn, dataOut = rd.npStormsFromFile("processedData/training.hdf5", 5, 15)
prediction = model.predict(dataIn)

for r, row in enumerate(prediction):
    print("----{}----".format(r))
    print("Pred: {}".format(row))
    print("Act:  {}".format(dataOut[r]))
    movie(dataIn[r], dataOut[r])