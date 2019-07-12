"""
ConvNet
Takes series of radar-images as input 
Predicts the categorisation of the next radar-image

using simple SGD optimizer because of https://arxiv.org/pdf/1705.08292.pdf
"""

import numpy as np
import tensorflow.keras as k
import radarData2 as rd
import datetime as dt
import time
import plotting as p
import os


batchSize = 4
nrTrainingSamples = 2000
nrValidationSamples = 50
timeSteps = int(5 * 60 / 5)
imageSize = 100
imageWidth = imageSize
imageHeight = imageSize
channels = 1


inpt_training, outpt_training = rd.tfDataGenerator("training_2017_0107_3007.h5", timeSteps, nrTrainingSamples)
inpt_validation, outpt_validation = rd.tfDataGenerator("validation_2016_0107_14_07.h5", timeSteps, nrValidationSamples)


model = k.models.Sequential([
    k.layers.Conv3D(1, (2, 2, 2), input_shape=(timeSteps, imageWidth, imageHeight, channels), name="conv1"),
    k.layers.Dropout(0.2),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(1, (2, 2, 2), name="conv2"),
    k.layers.Dropout(0.2),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(1, (2, 2, 2), name="conv3"),
    k.layers.Dropout(0.2),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(1, (2, 2, 2), name="conv4"),
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

# model = k.models.Sequential([
#     k.layers.Flatten(input_shape=(timeSteps, imageWidth, imageHeight, channels)),
#     k.layers.Dense(33, name="dense1", activation=k.activations.sigmoid),
#     k.layers.Dropout(0.2),
#     k.layers.Dense(15, name="dense2", activation=k.activations.sigmoid),
#     k.layers.Dropout(0.2),
#     k.layers.Dense(4, name="dense3", activation=k.activations.softmax)
# ])


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


history = model.fit(
    x=inpt_training,
    y=outpt_training,
    # validation_split=0.1,
    validation_data=(inpt_validation, outpt_validation),
    batch_size=3,
    epochs=8,
    callbacks=[modelSaver, tensorBoard, customPlotCallback]
)


tstp = int(time.time())
resultDir = "{}{}".format(tfDataDir, tstp)
if not os.path.exists(resultDir):
    os.makedirs(resultDir)
model.save("{}/simpleRadPredModel.h5".format(resultDir))
model.save("{}/latestRadPredModel.h5".format(tfDataDir))
p.createLossPlot("{}/loss.png".format(resultDir), history.history['loss'], history.history['val_loss'])


dataIn, dataOut = rd.loadTfData("validation_2016.h5", int(5 * 60 / 5), 20)
prediction = model.predict(dataIn)

for r, row in enumerate(prediction):
    print("----{}----".format(r))
    print("Pred: {}".format(row))
    print("Act:  {}".format(dataOut[r]))
