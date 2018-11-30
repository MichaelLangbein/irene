# ConvNet
# Takes series of radar-images as input 
# Predicts the categorisation of the next radar-image

import numpy as np
import tensorflow.keras as k
import radarData as rd
import datetime as dt



timeSteps = 15
fromTime = dt.datetime(2016, 9, 12)
toTime = dt.datetime(2016, 9, 15)
imageSize = 41
trainingData, trainingLabels = rd.getOverlappingLabeledTimeseries(imageSize, fromTime, toTime, timeSteps)
batchSize, timeSteps, imageWidth, imageHeight, channels = trainingData.shape


model = k.models.Sequential([
    k.layers.Conv3D(5, (2,2,2), input_shape=(timeSteps, imageWidth, imageHeight, 1), name="conv1"),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(5, (2,2,2), name="conv2"),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(5, (2,2,2), name="conv3"),
    k.layers.MaxPool3D(),
    k.layers.Flatten(),
    k.layers.Dense(33, name="dense1"),
    k.layers.Dense(10, name="dense2"),
    k.layers.Dense(3, name="dense3")
])


model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.mean_squared_error
)


model.fit(x=trainingData, y=trainingLabels, batch_size=batchSize, epochs=10)



