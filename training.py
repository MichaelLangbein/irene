import numpy as np
import tensorflow.keras as k
import data as rd
from data import rawDataDir, processedDataDir, frameHeight, frameWidth, frameLength
import datetime as dt
import time
import plotting as p
import os


batchSize = 4
stepsPerEpoch = int(2000 / batchSize)
validationSteps = int(200 / batchSize)
nrValidationSamples = 50
timeSteps = int(5 * 60 / 5)
imageSize = 100
imageWidth = imageSize
imageHeight = imageSize
channels = 1


training_generator = rd.DataGenerator(processedDataDir, dt.datetime(2016, 6, 1), dt.datetime(2016, 6, 30), batchSize, timeSteps)
validation_generator = rd.DataGenerator(processedDataDir, dt.datetime(2016, 7, 1), dt.datetime(2016, 7, 10), batchSize, timeSteps)


model = k.models.Sequential([
    k.layers.Conv3D(5, (2, 2, 2), input_shape=(timeSteps, imageWidth, imageHeight, channels), name="conv1"),
    k.layers.Dropout(0.2),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(15, (2, 2, 2), name="conv2"),
    k.layers.Dropout(0.2),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(5, (2, 2, 2), name="conv3"),
    k.layers.Dropout(0.2),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(5, (2, 2, 2), name="conv4"),
    k.layers.Dropout(0.2),
    k.layers.MaxPool3D(),
    # k.layers.Reshape((6, 121)),
    # k.layers.GRU(20, name="gru1"),
    k.layers.Flatten(),
    k.layers.Dense(33, name="dense1", activation=k.activations.sigmoid),
    k.layers.Dropout(0.2),
    k.layers.Dense(15, name="dense2", activation=k.activations.sigmoid),
    k.layers.Dropout(0.2),
    k.layers.Dense(4, name="dense3", activation=k.activations.softmax)
])


model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.categorical_crossentropy
)

print(model.summary())


thisDir = os.path.abspath('')
tfDataDir = thisDir + "/tfData/"

modelSaver = k.callbacks.ModelCheckpoint(
    tfDataDir + "simpleRadPredModel_checkpoint.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True
)

tensorBoard = k.callbacks.TensorBoard(
    log_dir='./tensorBoardLogs',
    histogram_freq=3,
    batch_size=32,
    write_graph=True,
    write_grads=False,
    write_images=False
)


class CustomPlotCallback(k.callbacks.Callback):
    losses = []
    vlosses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs["loss"])
        self.vlosses.append(logs["val_loss"])
        p.createLossPlot(tfDataDir + "/" + "loss.png", self.losses, self.vlosses)


customPlotCallback = CustomPlotCallback()


history = model.fit_generator(
    training_generator, 
    steps_per_epoch=stepsPerEpoch, 
    epochs=8, 
    verbose=2, 
    callbacks=[modelSaver, tensorBoard, customPlotCallback], 
    validation_data=validation_generator, 
    validation_steps=validationSteps
)


tstp = int(time.time())
resultDir = "{}{}".format(tfDataDir, tstp)
if not os.path.exists(resultDir):
    os.makedirs(resultDir)
model.save("{}/simpleRadPredModel.h5".format(resultDir))
model.save("{}/latestRadPredModel.h5".format(tfDataDir))
p.createLossPlot("{}/loss.png".format(resultDir), history.history['loss'], history.history['val_loss'])
