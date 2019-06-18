import os
import tensorflow.keras as k
import datetime as dt
import radarData2 as rd
import numpy as np
import plotting as pl


thisDir = os.path.dirname(os.path.abspath(__file__))
tfDataDir = thisDir + "/tfData/"
modelName = "latestRadPredModel.h5"


model = k.models.load_model(tfDataDir + modelName)
dataIn, dataOut = rd.loadTfData("validation_2016.h5", int(5 * 60/5), 10)
prediction = model.predict(dataIn)


maxActInpt = pl.getMaximallyActivatingImage(model, "conv2", 0, dataIn[0].shape)
pl.movie(maxActInpt[:, :, :, 0], ["maximal activationImage for conv2 channel 0 "], 100)

maxActInpt = pl.getMaximallyActivatingImage(model, "conv3", 0, dataIn[0].shape)
pl.movie(maxActInpt[:, :, :, 0], ["maximal activationImage for conv3 channel 0 "], 100)

pl.plotActivations(model, dataIn[6], "conv3")



for r, row in enumerate(prediction):
    print("----{}----".format(r))
    print("Pred: {}".format(row))
    print("Act:  {}".format(dataOut[r]))
    if np.random.rand() > 0.95:
        pl.movie(dataIn[r, :, :, :, 0], dataOut[r], 15)