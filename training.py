import numpy as np
import tensorflow.keras as k
from data import Film, Frame, analyseAndSaveTimeRange, DataGenerator
from config import rawDataDir, processedDataDir, tfDataDir
import datetime as dt
import time
import plotting as p
import os


batchSize = 10
nrBatchesPerEpoch = 100
nrEpochs = 20
validationSteps = 10
nrValidationSamples = 50
timeSteps = int(5 * 60 / 5)
imageSize = 100
imageWidth = imageSize
imageHeight = imageSize
channels = 1


# creating trainingdata
trainingStart = dt.datetime(2016, 6, 1)
trainingEnd = dt.datetime(2016, 6, 30)
validationStart = dt.datetime(2016, 7, 1)
validationEnd = dt.datetime(2016, 7, 15)

# getting generators
training_generator = DataGenerator(processedDataDir, trainingStart, trainingEnd, nrBatchesPerEpoch, batchSize, timeSteps)
validation_generator = DataGenerator(processedDataDir, validationStart, validationEnd, nrBatchesPerEpoch, batchSize, timeSteps)


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
    loss=k.losses.categorical_crossentropy, 
    metrics=[k.metrics.categorical_accuracy]
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
    accs = []
    vaccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs["loss"])
        self.vlosses.append(logs["val_loss"])
        p.createLossPlot(tfDataDir + "/" + "loss.png", self.losses, self.vlosses)
        self.accs.append(logs["categorical_accuracy"])
        self.vaccs.append(logs["val_categorical_accuracy"])
        p.createLossPlot(tfDataDir + "/" + "acc.png", self.accs, self.vaccs)


customPlotCallback = CustomPlotCallback()


history = model.fit_generator(
    training_generator,
    epochs=nrEpochs,
    verbose=2,
    callbacks=[modelSaver, tensorBoard, customPlotCallback],
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=4, 
)


tstp = int(time.time())
resultDir = "{}{}".format(tfDataDir, tstp)
if not os.path.exists(resultDir):
    os.makedirs(resultDir)
model.save("{}/simpleRadPredModel.h5".format(resultDir))
model.save("{}/latestRadPredModel.h5".format(tfDataDir))
p.createLossPlot("{}/loss.png".format(resultDir), history.history['loss'], history.history['val_loss'])
p.createLossPlot("{}/accuracy.png".format(resultDir), history.history['categorical_accuracy'], history.history['val_categorical_accuracy'])