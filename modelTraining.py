# ConvNet
# Takes series of radar-images as input 
# Predicts the categorisation of the next radar-image

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


inpt_training, outpt_training = rd.loadTfData("training_2016.h5", timeSteps, 100)
print(f"input: {np.sum(outpt_training, axis=0)}")


model = k.models.Sequential([
    k.layers.Conv3D(5, (2,2,2), input_shape=(timeSteps, imageWidth, imageHeight, channels), name="conv1"),
    k.layers.MaxPool3D(),
    k.layers.Conv3D(5, (2,2,2), name="conv2"),
    k.layers.MaxPool3D(),
    k.layers.Flatten(),
    k.layers.Dense(33, name="dense1", activation=k.activations.sigmoid),
    k.layers.Dense(15, name="dense2", activation=k.activations.sigmoid),
    k.layers.Dense(3, name="dense5", activation=k.activations.softmax)
])


model.compile(
    optimizer=k.optimizers.SGD(lr=0.05, clipvalue=0.5),
    loss=k.losses.categorical_crossentropy
)

print(model.summary())


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
        createLossPlot(tfDataDir + "/" + "loss.png", self.losses, self.vlosses)
        

customPlotCallback = CustomPlotCallback()



# history = model.fit_generator(
#     generator=trainingGenerator,
#     steps_per_epoch=25,       # number of batches to be drawn from generator
#     epochs=10,                 # number of times the data is repeated
#     validation_data=validationGenerator,
#     validation_steps=5,       # number of batches to be drawn from generator
#     callbacks=[modelSaver, tensorBoard, customPlotCallback] 
# )

history = model.fit(
    x=inpt_training, 
    y=outpt_training, 
    validation_split=0.1, 
    batch_size=3, 
    epochs=20, 
    callbacks=[modelSaver, tensorBoard, customPlotCallback]
)


tstp = int(t.time())
resultDir = "{}{}".format(tfDataDir, tstp)
if not os.path.exists(resultDir):
    os.makedirs(resultDir)
model.save("{}/simpleRadPredModel.h5".format(resultDir))
model.save("{}/latestRadPredModel.h5".format(tfDataDir))
createLossPlot("{}/loss.png".format(resultDir), history.history['loss'], history.history['val_loss'])



dataIn, dataOut = rd.loadTfData("training_2016.h5", int(5 * 60/5), 20)
prediction = model.predict(dataIn)

for r, row in enumerate(prediction):
    print("----{}----".format(r))
    print("Pred: {}".format(row))
    print("Act:  {}".format(dataOut[r]))