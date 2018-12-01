# ConvNet
# Takes series of radar-images as input 
# Predicts the categorisation of the next radar-image

import numpy as np
import tensorflow.keras as k
import radarData as rd
import datetime as dt



batchSize = 20
timeSteps = 15
imageSize = 41
genFac = rd.GeneratorFactory(batchSize, timeSteps, imageSize)
trainingGenerator = genFac.createGenerator()
validationGenerator = genFac.createGenerator()
batchSize, timeSteps, imageWidth, imageHeight, channels = genFac.getDimensions()


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

history = model.fit_generator(
    generator=trainingGenerator(),
    steps_per_epoch=30,       # number of batches to be drawn from generator
    epochs=3,                 # number of times the data is repeated
    validation_data=validationGenerator(),
    validation_steps=30       # number of batches to be drawn from generator
)

