import os
import tensorflow.keras as k
import datetime as dt
import data as rd
from data import rawDataDir, processedDataDir, frameHeight, frameWidth, frameLength, Film, Frame, analyseAndSaveTimeRange
import numpy as np
import plotting as pl
import matplotlib.pyplot as plt


print("----starting up-----")
thisDir = os.path.abspath('')
tfDataDir = thisDir + "/tfData/"
modelName = "latestRadPredModel.h5"
batchSize = 4
stepsPerEpoch = int(2000 / batchSize)
validationSteps = int(200 / batchSize)
nrValidationSamples = 50
timeSteps = int(5 * 60 / 5)
imageSize = 100
imageWidth = imageSize
imageHeight = imageSize
channels = 1



model = k.models.load_model(tfDataDir + modelName)
generator = rd.DataGenerator(processedDataDir, dt.datetime(2016, 6, 1), dt.datetime(2016, 6, 3), batchSize, timeSteps)

# maxActInpt = pl.getMaximallyActivatingImage(model, "conv4", 0, dataIn[0].shape)
# pl.showMovie(maxActInpt[:, :, :, 0], ["maximal activationImage for conv4 channel 0 "], 100)


def getMaxIndex(list):
    return np.where(list == np.amax(list))


print("----predicting----")
i = 0
for dataIn, dataOut in generator: 
    predictions = model.predict(dataIn)
    for r in range(len(predictions)):
        target = dataOut[r]
        maxIndexTarget = getMaxIndex(target)
        prediction = predictions[r]
        maxIndexPrediction = getMaxIndex(prediction)
        print(f"target: {target}")
        print(f"prediction: {prediction}")
        print(f"correctly predicted {maxIndexPrediction == maxIndexTarget}")
        pl.showActivation(model, dataIn[r], "conv3", 2)
    i += 1
    if i > 1:
        break