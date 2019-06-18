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
import time as t
import matplotlib.pyplot as plt
import os


thisDir = os.path.dirname(os.path.abspath(__file__))
tfDataDir = thisDir + "/tfData/"

batchSize = 4
timeSteps = int(5 * 60/5)
imageSize = 100
imageWidth = imageSize
imageHeight = imageSize
channels = 1


inpt_training, outpt_training = rd.loadTfData("training_2016.h5", timeSteps, 150)
inpt_validation, outpt_validation = rd.loadTfData("validation_2016.h5", timeSteps, 50)


model = k.models.Sequential([
    k.layers.Conv3D(5, (2,2,2), input_shape=(timeSteps, imageWidth, imageHeight, channels), name="conv1"),
    k.layers.Dropout(0.2),
    k.layers.MaxPool3D(),
    k.layers.Flatten(),
    k.layers.Dense(20, name="dense1", activation=k.activations.sigmoid),
    k.layers.Dropout(0.2),
    k.layers.Dense(4, name="dense7", activation=k.activations.softmax)
])


model.compile(
    optimizer=k.optimizers.SGD(lr=0.05, clipvalue=0.5),
    loss=k.losses.categorical_crossentropy
)

print(model.summary())


modelSaver = k.callbacks.ModelCheckpoint(
    tfDataDir + "smallerRadPredModel_checkpoint.h5", 
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


def createLossPlot(filePath, loss, vloss):
    plt.plot(loss)
    plt.plot(vloss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(filePath)


class CustomPlotCallback(k.callbacks.Callback):
    losses = []
    vlosses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs["loss"])
        self.vlosses.append(logs["val_loss"])
        createLossPlot(tfDataDir + "/" + "smallerModel_loss.png", self.losses, self.vlosses)
        

customPlotCallback = CustomPlotCallback()


history = model.fit(
    x=inpt_training, 
    y=outpt_training, 
    #validation_split=0.1, 
    validation_data=(inpt_validation, outpt_validation),
    batch_size=3, 
    epochs=8, 
    callbacks=[modelSaver, tensorBoard, customPlotCallback]
)


tstp = int(t.time())
resultDir = "{}{}".format(tfDataDir, tstp)
if not os.path.exists(resultDir):
    os.makedirs(resultDir)
model.save("{}/smallerRadPredModel.h5".format(resultDir))
model.save("{}/latestsmallerRadPredModel.h5".format(tfDataDir))
createLossPlot("{}/smallerModel_eloss.png".format(resultDir), history.history['loss'], history.history['val_loss'])



dataIn, dataOut = rd.loadTfData("validation_2016.h5", int(5 * 60/5), 20)
prediction = model.predict(dataIn)

for r, row in enumerate(prediction):
    print("----{}----".format(r))
    print("Pred: {}".format(row))
    print("Act:  {}".format(dataOut[r]))