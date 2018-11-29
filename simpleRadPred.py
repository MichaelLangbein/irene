# ConvNet
# Takes series of radar-images as input 
# Predicts the catigorisation of the next radar-image

import numpy as np
import tensorflow.keras as k
import radarData as rd
import datetime as dt



timeSteps = 7
fromTime = dt.datetime(2016, 9, 1)
toTime = dt.datetime(2016, 9, 30)
imageSize = 20
trainingData, trainingLabels = rd.getOverlappingLabeledTimeseries(imageSize, fromTime, toTime, timeSteps)
batchSize, timeSteps, imageWidth, imageHeight, channels = trainingData.shape


model = k.models.Sequential(
    k.layers.Conv3D(5, (2,2,2), input_shape=(timeSteps, imageWidth, imageHeight, 1)),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(5, (2,2,2)),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(5, (2,2,2)),
    k.layers.MaxPool3D(),
    k.layers.Dense(33),
    k.layers.Dense(10),
    k.layers.Dense(3)
)


model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.mean_squared_error
)


model.fit(x=trainingData, y=trainingLabels, batch_size=batchSize, epochs=10)



