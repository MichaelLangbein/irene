import numpy as np
import tensorflow.keras as k
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import wradlib as wrl

"""
    understanding matplotlib

        general workflow

            fig, axesArr = plt.sublplots(2, 2)
            doFirstPlot(axesArr[0])
            doSecondPlot(axesArr[1])
            ...
            plt.show()

"""



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


def plotActivations(model, inputSample, layerName):

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

    allData = []
    for t in range(T):
        row = []
        for c in range(C):
            data = activation[0, t, :, :, c]
            row.append(data)
        allData.append(row)

    fig, axArr = plotGrids(allData, np.max(activation))

    for t in range(T):
        for c in range(C): 
            axArr[t, c].set_title(f"time {t} channel {c}")

    fig.suptitle(f"layer {layerName}")
    return fig, axArr


def plotGrids(allData: list, maxV=500):
    nrRows = len(allData)
    nrCols = len(allData[0])
    fig, axesArr = plt.subplots(nrRows, nrCols)
    axesArr = np.reshape(axesArr, (nrRows, nrCols))
    for r in range(nrRows):
        for c in range(nrCols):
            plotGrid(axesArr[r, c], allData[r][c], maxV)
    return fig, axesArr


def plotGrid(axes, data, vmax):
    img = axes.imshow(data)
    img.norm.vmin = 0
    img.norm.vmax = vmax


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


def movie(data: np.array, labels, interval=500):

    fig = plt.figure()

    img = plt.imshow(data[0])
    img.norm.vmin = np.min(data)
    img.norm.vmax = np.max(data)
    axes = fig.get_axes()

    labelsString = ", ".join([str(label) for label in labels])
    axes[0].set_title(labelsString)

    def animate(frameNr):
        frame = data[frameNr]
        img.set_data(frame)
        plt.xlabel("Frame {},  maxval {}".format(frameNr, np.max(frame)))
        return img, labelsString

    animation = FuncAnimation(fig, animate, frames=range(data.shape[0]), interval=interval, repeat=True, repeat_delay=1000)

    plt.show()
