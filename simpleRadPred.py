# ConvNet
# Takes series of radar-images as input 
# Predicts the categorisation of the next radar-image

import numpy as np
import tensorflow.keras as k
import radarData as rd
import time as t
import matplotlib.pyplot as plt
import os


thisDir = os.path.dirname(os.path.abspath(__file__))
tfDataDir = thisDir + "/tfData/"

batchSize = 20
timeSteps = 15
imageSize = 41
imageWidth = imageSize
imageHeight = imageSize
channels = 1
trainingGenerator = rd.radarGenerator(batchSize, timeSteps, imageSize)
validationGenerator = rd.radarGenerator(batchSize, timeSteps, imageSize)


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


checkpoint = k.callbacks.ModelCheckpoint(
    tfDataDir + "simpleRadPredModel_checkpoint.h5", 
    monitor="val_loss", 
    verbose=1, 
    save_best_only=True, 
    mode="min"
)


history = model.fit_generator(
    generator=trainingGenerator,
    steps_per_epoch=15,       # number of batches to be drawn from generator
    epochs=3,                 # number of times the data is repeated
    validation_data=validationGenerator,
    validation_steps=3,       # number of batches to be drawn from generator
    callbacks=[checkpoint]
)


tstp = t.time()
model.save(tfDataDir + "simpleRadPredModel_{}.h5".format(tstp))


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(tfDataDir + "loss_{}.png".format(tstp))
