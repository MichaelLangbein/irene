import numpy as np
import tensorflow.keras as k
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import wradlib as wrl
from typing import Callable

"""
    understanding matplotlib

        general workflow

            fig, axesArr = plt.sublplots(2, 2)
            doFirstPlot(axesArr[0])
            doSecondPlot(axesArr[1])
            ...
            plt.show()

"""



def showActivation(model, inputSample, layerName, channel):

    # We turn the one input into a batch of size one
    T, H, W, C = inputSample.shape
    inputBatch = np.reshape(inputSample, (1, T, H, W, C))

    # We create the model
    activationModel = k.models.Model(
        inputs=[model.input], 
        outputs=[model.get_layer(layerName).output]
    )

    activation = activationModel.predict(inputBatch)
    N, T, H, W, C = activation.shape
    sampleActivation = activation[0]

    fig, axes = plt.subplots(2)
    
    img1 = plt.imshow(inputSample[0, :, :, 0])
    img1.norm.vmin = np.min(inputSample)
    img1.norm.vmax = np.max(inputSample)

    def animate1(frameNr):
        img1.set_data(inputSample[frameNr, :, :, 0])
        return img1
    
    animation1 = FuncAnimation(fig, animate1, frames=range(T), interval=15, repeat=True, repeat_delay=1000)

    img2 = plt.imshow(sampleActivation[0, :, :, channel])
    img2.norm.vmin = np.min(sampleActivation)
    img2.norm.vmax = np.max(sampleActivation)

    def animate2(frameNr):
        img2.set_data(sampleActivation[frameNr, :, :, channel])
        return img2
    
    animation2 = FuncAnimation(fig, animate2, frames=range(T), interval=15, repeat=True, repeat_delay=1000)

    plt.show()


def showActivations(model, inputSample: np.array, layerName):
    # We turn the one input into a batch of size one
    T, H, W, C = inputSample.shape
    inputSample = np.reshape(inputSample, (1, T, H, W, C))

    # We create the model
    activationModel = k.models.Model(
        inputs=[model.input], 
        outputs=[model.get_layer(layerName).output]
    )

    activation = activationModel.predict(inputSample)
    N, T, H, W, C = activation.shape

    figure, axArr = plt.subplots(C)
    for c in range(C):
        movie(figure, axArr[c], activation[0, :, :, :, c], [f"activation for layer {layerName}, channel {c}"])

    plt.show()




def getMaximallyActivatingImage(model, layerName: str, channelNr: int, imageDimensions: tuple):

    # we build a model
    layerOutput = model.get_layer(layerName).output
    lossFunc = K.mean(layerOutput[:, :, :, :, channelNr])
    gradFunc = K.gradients(lossFunc, model.input)[0]
    gradFunc /= (K.sqrt(K.mean(K.square(gradFunc))) + 1e-5)
    tap = K.function([model.input], [lossFunc, gradFunc])

    # we create a random input-image
    image = np.random.random((1,) + imageDimensions)

    # we optimize the image so that it activates the channel maximally
    delta = 0.8
    for t in range(40):
        loss_val, grad_val = tap([image])
        image += grad_val * delta

    return image[0]




def plotGrids(allData: list, plotFunc: Callable):
    nrRows = len(allData)
    nrCols = len(allData[0])
    fig, axesArr = plt.subplots(nrRows, nrCols)
    axesArr = np.reshape(axesArr, (nrRows, nrCols))
    for r in range(nrRows):
        for c in range(nrCols):
            plotFunc(fig, axesArr[r, c], allData[r][c])
    return fig, axesArr


def plotRadolanFrames(films, rows, cols, time):
    fig, axesArr = plt.subplots(rows, cols)
    for r in range(rows):
        for c in range(cols):
            film = films[r*cols + c]
            frame = film.frames[time]
            plotGrid(axesArr[r, c], frame.data, 500)
    plt.show()


def createLossPlot(filePath, loss, vloss):
    plt.plot(loss)
    plt.plot(vloss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(filePath)


def plotRadolanData(axes, data, attrs, clabel=None):
    grid = wrl.georef.get_radolan_grid(*data.shape)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, aspect='equal')
    x = grid[:,:,0]
    y = grid[:,:,1]
    pm = ax.pcolormesh(x, y, data, cmap='viridis')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label(clabel)
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.title('{0} Product\n{1}'.format(attrs['producttype'], attrs['datetime'].isoformat()))
    plt.xlim((x[0,0],x[-1,-1]))
    plt.ylim((y[0,0],y[-1,-1]))
    plt.grid(color='r')
    plt.show()


def movie(fig, axes, data: np.array, labels=[], interval=500):
    img = plt.imshow(data[0])
    img.norm.vmin = np.min(data)
    img.norm.vmax = np.max(data)

    labelsString = ", ".join([str(label) for label in labels])
    axes.set_title(labelsString)

    def animate(frameNr):
        frame = data[frameNr]
        img.set_data(frame)
        plt.xlabel("Frame {},  maxval {}".format(frameNr, np.max(frame)))
        return img, labelsString

    animation = FuncAnimation(fig, animate, frames=range(data.shape[0]), interval=interval, repeat=True, repeat_delay=1000)
    return animation


def showMovie(data: np.array, labels, interval=500):
    figure, axes = plt.subplots(1)
    animation = movie(figure, axes, data, labels, interval)
    plt.show()
