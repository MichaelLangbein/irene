import numpy as np
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


def plotGrids(allData):
    nrRows = len(allData)
    nrCols = len(allData[0])
    fig, axesArr = plt.subplots(nrRows, nrCols)
    for r in range(nrRows):
        for c in range(nrCols):
            plotGrid(axesArr[r, c], allData[r][c], 500)
    plt.show()


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


def movie(data: np.array, labels, interval=500, saveFileName = None):

    fig = plt.figure()

    img = plt.imshow(data[0])
    img.norm.vmin = np.min(data)
    img.norm.vmax = np.max(data)
    axes = fig.get_axes()

    labelsString = ", ".join(str(labels))
    axes[0].set_title(labelsString)

    def animate(frameNr):
        frame = data[frameNr]
        img.set_data(frame)
        plt.xlabel("Frame {},  maxval {}".format(frameNr, np.max(frame)))
        return img, labelsString

    animation = FuncAnimation(fig, animate, frames=range(data.shape[0]), interval=interval, repeat=True, repeat_delay=1000)

    if not saveFileName: 
        plt.show()
    else:
        animation.save(saveFileName + ".mp4", writer='ffmpeg')
